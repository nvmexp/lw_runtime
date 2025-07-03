/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ptx.l
 *
 *  Description              :
 *
 */

%{

/*--------------------------------- Includes ---------------------------------*/

#include "ptx.h"
#include "ptx_parser.h"
#include "ptxparseMessageDefs.h"
#include "stdLocal.h"
#include "stdMessages.h"
#include "stdString.h"
#include "ctMessages.h"
#include "ctLog.h"

// OPTIX_HAND_EDIT
#include <stdbool.h>
#include <stdint.h>

/*------------------------------ Definitions ---------------------------------*/

/*
* Override value of YY_BUF_SIZE to a bigger value. 
* The default value is not enough
* for files having long comments.
* Have undef followed by def sequence
* to avoid compiler warning.
*/
#undef YY_BUF_SIZE
#define YY_BUF_SIZE 32768


//>> OPTIX_HAND_EDIT
// adding additional state data to allow streaming decryption of pushedInput
#define OPTIX_DECRYPTED_BUFFER_SIZE 256
typedef struct OptixInputStateRec {
    char decryptedBuffer[OPTIX_DECRYPTED_BUFFER_SIZE];
    size_t decryptedBufferLen;    /* counts down from OPTIX_DECRYPTED_BUFFER_SIZE to 0 */
    size_t ptxInputLen;           /* matches when we fill up the buffer */
    size_t lwrrentBufferSize;     /* matches usually OPTIX_DECRYPTED_BUFFER_SIZE unless in last "round" */
    void* decrypter;              /* the EncryptionManager instance */
    GenericCallback decryptCall;  /* the C interface function to trigger decryption */
} OptixInputStateRec;
typedef OptixInputStateRec *OptixInputState; 
//<< OPTIX_HAND_EDIT

stdUNUSED( static void  yyunput( int c, register char *yy_bp, yyscan_t yyscanner ) );
/*
 * Function        : Used to initialize the ptxfilename and ptxlineno and add to include stack of ptxFileSourceStruct
 * Parameters      : gblState     (I) Current ptx parsing state. 
 *                 : newFileName  (I) name of a file which is about to be parsed
 *                   newLineNo    (I) line number of the file from which parsing would start
 *                                    This is always 1.
 * Note            : Affects ptxlineno and ptxfilename as a side effect
 */
void ptxInitLwrrentSourcePos(ptxParsingState gblState, String newFileName, uInt newLineNo)
{
    msgPushInclude( gblState->ptxFileSourceStruct, ptxget_lineno(gblState->scanner), newFileName);
    gblState->parseData->ptxfilename = newFileName;
    ptxset_lineno(newLineNo, gblState->scanner);
}

/*
 * Function        : Used to push a new input file/string for parsing and add to include stack of ptxFileSourceStruct
 * Parameters      : gblState     (I) Current ptx parsing state. 
 *                 : newFileName  (I) name of a file / desc. of the string which is about to be parsed
 *                   newLineNo    (I) line number of the file from which parsing would start
 * Note            : Affects ptxlineno and ptxfilename as a side effect
 */
void ptxPushSourcePos(ptxParsingState gblState, cString newFileName, uInt newLineNo)
{
    msgSourcePos_t lwrrentSourcePos= ptxsLwrPos(gblState);

    listAddTo( stdCOPY(&lwrrentSourcePos), &gblState->parseData->pushedSourcePos );
    msgPushInclude( gblState->ptxFileSourceStruct, ptxget_lineno(gblState->scanner), newFileName);

    gblState->parseData->ptxfilename = newFileName;
    ptxset_lineno(newLineNo, gblState->scanner);
}


/*
 * Function        : Pop the top most file from the file include stack pushedSourcePos and JAS's ptxFileSourceStruct
 * Parameters      : gblState     (I) Current ptx parsing state. 
 * Note            : Affects ptxlineno and ptxfilename as a side effect
 */
static void restoreSourcePos(ptxParsingState gblState)
{
    stdList_t       l   = gblState->parseData->pushedSourcePos;
    msgSourcePos_t* top =  ((msgSourcePos_t*)gblState->parseData->pushedSourcePos->head);

    gblState->parseData->ptxfilename= msgGetFileName(*top);
    ptxset_lineno((*top)->lineNo, gblState->scanner);

    gblState->parseData->pushedSourcePos   = gblState->parseData->pushedSourcePos->tail;
    stdFREE(l);


    msgPopInclude(gblState->ptxFileSourceStruct);
}

msgSourcePos_t ptxsLwrPos(ptxParsingState gblState)
{
    msgSourcePos_t result= ctMsgCreateSourcePos( gblState->parseData->ptxfilename, &(gblState->ptxFileSourceStruct), ptxget_lineno(gblState->scanner) );

    return result;
}


/*
 * Function        : Distinguish whether a file is user input or an internal/dummy file
 * Parameters      : fileName  (I) name of a file whose type is to be determined
 * Function Result : True, if filename is name of a user input file
 *                   Else, False
 * Note            : Internal files like macro expansions begin with '<'
 *                   In JIT path, filename is empty string "". As set in ptxParseInputString.
 */
Bool isPtxUserInput(cString fileName)
{
   return (fileName && fileName[0] != '<' && fileName[0] != '\0');
}

/*------------------------------ Definitions ---------------------------------*/

static int my_input_blkread(ptxParsingState gblState, String buf, int max_size,Bool expandMacro, Bool incrementLineNo);

static int my_input(ptxParsingState gblState, String buf, Bool expandMacro, Bool incrementLineNo)
{
    return my_input_blkread(gblState, buf, 0, expandMacro, incrementLineNo);
}


void ptxDefinePreprocessorMacro(ptxParsingState gblState, String name, String value)
{
    mapDefine( gblState->preprocessorMacros, name, value );
}

#define YY_INPUT(buf,result,max_size) \
           result = my_input_blkread((ptxParsingState) ptxget_extra(yyscanner), buf,max_size,True,False);


/*------------------------------ Definitions ---------------------------------*/


msgSourcePos_t ptxUserPos(ptxParsingState gblState, msgSourcePos_t *origPos)
{
    stdList_t pos = gblState->parseData->pushedSourcePos;
    String lFileName;
    if (pos) {
        msgSourcePos_t *p;
        while (pos->tail) {
            String fileName = msgGetFileName(*((msgSourcePos_t*)(pos->tail->head)));
            if (fileName == NULL || fileName[0] == '\0') {
                break;
            }
            pos = pos->tail;
        }
        p = (msgSourcePos_t*)(pos->head);
        lFileName = msgGetFileName(*p);
        if (isPtxUserInput(lFileName)) {
            return *p;
        }
    }
    return *origPos;
}

// TODO: Evaluate if we need to implement custom  my_input_blkread or we can use auto generated
//       by flex.
static int my_input_blkread(ptxParsingState gblState, String buf, int max_size,Bool expandMacro, Bool incrementLineNo)
{
    Char c= 0;
    
    if (gblState->parseData->ptxLookahead) {
        c = gblState->parseData->ptxLookahead;
        gblState->parseData->ptxLookahead= 0;
    } else {
    
        if (gblState->parseData->pushedInput) { 
          // OPTIX_HAND_EDIT
          // if decryption is enabled (==optixInputState) then pushedInput needs to be decrypted into
          // optixInputState->decryptedBuffer before further consumption. As decoding might consume
          // more bytes then are added to the decryptedBuffer one needs to keep track of numConsumed too.
          if ((OptixInputState)gblState->optixInputState) {
              if (!((OptixInputState)gblState->optixInputState)->decryptedBufferLen) {
                  typedef bool(*DecryptStringCall)(void*, const char*, size_t, char*, size_t*, size_t, size_t*);
                  size_t numConsumed;
                  bool res = (*((DecryptStringCall)((OptixInputState)gblState->optixInputState)->decryptCall))(
                      ((OptixInputState)gblState->optixInputState)->decrypter,
                      gblState->parseData->pushedInput,
                      gblState->parseData->pushedInputLen,
                      ((OptixInputState)gblState->optixInputState)->decryptedBuffer,
                      &((OptixInputState)gblState->optixInputState)->decryptedBufferLen,
                      OPTIX_DECRYPTED_BUFFER_SIZE,
                      &numConsumed
                  );

                  gblState->parseData->pushedInput += numConsumed;
                  gblState->parseData->pushedInputLen -= numConsumed;
                  // the current buffer size is usually equal to OPTIX_DECRYPTED_BUFFER_SIZE. But during the
                  // last refill most probably < OPTIX_DECRYPTED_BUFFER_SIZE bytes are decrypted. Hence we need
                  // to keep track of the current buffer size to retrieve each character c correctly.
                  ((OptixInputState)gblState->optixInputState)->lwrrentBufferSize = ((OptixInputState)gblState->optixInputState)->decryptedBufferLen;
              }
              if (((OptixInputState)gblState->optixInputState)->decryptedBufferLen) {
                  c = ((OptixInputState)gblState->optixInputState)->decryptedBuffer[((OptixInputState)gblState->optixInputState)->lwrrentBufferSize - ((OptixInputState)gblState->optixInputState)->decryptedBufferLen];
                  --((OptixInputState)gblState->optixInputState)->decryptedBufferLen;
              }
              else
                  c = 0;
          }
          else {
              if (gblState->parseData->pushedInputLen > 0) {
                  c = *(gblState->parseData->pushedInput++); 
                  --gblState->parseData->pushedInputLen;
              }
              else
                  c = 0;
          }

          if (gblState->ptxObfuscation) {
              if (gblState->ptxLength) {
                  ++gblState->parseData->ptxCount;
                  if (gblState->parseData->ptxCount >= gblState->ptxLength && c == 0) {
                      /* Obfuscation ends up putting 0 where not end of string,
                       * so instead use ptxLength to know where to stop. */
                      gblState->parseData->ptxCount = 0; // reset
                  } else {
                      c= stdDeobfuscate(gblState->ptxObfuscation,c);
                  }
              } else {
                  /* Obfuscation also used with macros which do not set length
                   * but happen to be okay. */
                  c= stdDeobfuscate(gblState->ptxObfuscation,c);
              }
          }

          if (c == 0) { 
              String     s  = gblState->parseData->pushedInputs->head;
              stdList_t  l1 = gblState->parseData->pushedInputs;
              // OPTIX_HAND_EDIT
              stdList_t  l4 = gblState->parseData->pushedInputLens;
              stdList_t  l2 = gblState->parseData->pushedPending;
              stdList_t  l3 = gblState->parseData->pushedObfuscators;

              restoreSourcePos(gblState);
              
              if (gblState->ptxObfuscation) {
                  stdDeleteObfuscation(gblState->ptxObfuscation);
              }
              
              gblState->parseData->pushedInputs      = gblState->parseData->pushedInputs->tail;
              // OPTIX_HAND_EDIT
              gblState->parseData->pushedInputLens   = gblState->parseData->pushedInputLens->tail;
              gblState->parseData->pushedPending     = gblState->parseData->pushedPending->tail;
              gblState->parseData->pushedObfuscators = gblState->parseData->pushedObfuscators->tail;

              stdFREE(l1);
              stdFREE(l2);
              stdFREE(l3);
              // OPTIX_HAND_EDIT
              stdFREE(l4);

              gblState->parseData->pushedInput= NULL;
              // OPTIX_HAND_EDIT
              gblState->parseData->pushedInputLen = 0;

             /*
              * Decide if there is any more input:
              */
              if (!gblState->parseData->pushedInputs && gblState->ptxStringInput) {
                  c= 0;
              } else {
                  stdFREE(s);
                  if (gblState->parseData->pushedInputs) {
                      gblState->ptxObfuscation = gblState->parseData->pushedObfuscators->head;
                      gblState->parseData->pushedInput    = gblState->parseData->pushedPending->head;
                  }
                  return my_input_blkread(gblState, buf,max_size,expandMacro,incrementLineNo);
              }
          }
        } else {
          if (max_size != 0) {
              int count = 0;
              if (gblState->ptxObfuscation && gblState->ptxLength) {
                  /* Obfuscated string may have EOF char, but not real EOF,
                   * so read exact number of bytes. */
                  for (; count < max_size && count < gblState->ptxLength; ++count) {
                      buf[count] = (char)fgetc(gblState->ptxin);
                  }
              } else {
                  count = fread(buf, sizeof(Char), max_size, gblState->ptxin);
              }

              if (count == 0) { 
                  return 0;
              } else { 
                  if (gblState->ptxObfuscation) { 
                      int i;
                      // obfuscated code is not proper string,
                      // so use memcpy not strcpy
                      String bufcpy = stdCOPY_S(buf, count);
                      for (i=0; i< count; i++)
                        buf[i]= stdDeobfuscate(gblState->ptxObfuscation,bufcpy[i]);
                      stdFREE(bufcpy);
                  }
                  return count;
              }
          } else {
              Int cc = fgetc(gblState->ptxin);
              if (cc == EOF) { 
                  c= 0;
              } else {
                  if (gblState->ptxObfuscation) {
                      c= stdDeobfuscate(gblState->ptxObfuscation,cc);
                  } else {
                      c= cc;
                  }
              }
          }
        }
    }
    
    if (expandMacro && c == '$') { 
        my_input(gblState, &c,True,incrementLineNo);
         
        if (c != '{') { 
            gblState->parseData->ptxLookahead = c ;
            c            = '$';
        } else { 
            stdString_t    s  = stringNEW();
            cString spFileName =  gblState->parseData->ptxfilename;
            uInt   spLineNum   =  ptxget_lineno(gblState->scanner);
            String         text;
            String         macroName;

            while (my_input(gblState, &c,True,incrementLineNo) && c != '}') {
                stringAddChar(s,c);
            }

            stdCHECK_WITH_POS( c=='}', (ptxMsgNonterminatedPreprMacro, ptxsLwrPos(gblState)) );

            macroName = stringStripToBuf(s);
            text      = mapApply( gblState->preprocessorMacros, macroName );

            stdCHECK_WITH_POS( text, (ptxMsgUnknownPreprMacro, ptxsLwrPos(gblState), macroName) ) {
                // OPTIX_HAND_EDIT
                ptxPushInput( gblState, stdCOPYSTRING(text), strlen(text), NULL, spFileName, spLineNum, 0, 0);
                stdFREE     ( macroName );
            }

            return my_input_blkread(gblState, buf,max_size,True,incrementLineNo);
        }
    }
    
    if (incrementLineNo && c=='\n') {
        ptxset_lineno(ptxget_lineno(gblState->scanner) + 1,
                      gblState->scanner);
    }
    
   *buf= c;
   return c != 0;
}

// OPTIX_HAND_EDIT
void ptxPushInput(ptxParsingState gblState, String s, uInt32 sLen, stdObfuscationState state, cString newFileName, uInt newLineNo, void* decrypter, GenericCallback decryptionCB )
{
    if (gblState->parseData->pushedInput) {
        gblState->parseData->pushedPending->head= gblState->parseData->pushedInput;
    }
    
    listAddTo( s,                        &gblState->parseData->pushedInputs      );
    // OPTIX_HAND_EDIT
    listAddTo( (Pointer) ((intptr_t) sLen), &gblState->parseData->pushedInputLens   );
    listAddTo( s,                        &gblState->parseData->pushedPending     );
    listAddTo( state,                    &gblState->parseData->pushedObfuscators );
    
    gblState->parseData->pushedInput    = s;
    // OPTIX_HAND_EDIT
    gblState->parseData->pushedInputLen = sLen;
    
    gblState->ptxObfuscation = state;
    // OPTIX_HAND_EDIT
    // when encryption is enabled, a decrypter and a decryption callback are passed to the lexer.
    // To keep this and additionally required data an optixInputState instance is allocated and
    // filled accordingly.
    if( decrypter )
    {
        // imitating behavior of "stdNEW( (OptixInputState)gblState->optixInputState );" which breaks compilation
        gblState->optixInputState = (void*)( (OptixInputStateRec*)__MALLOC( sizeof( OptixInputStateRec ) ) );
        if( !gblState->optixInputState )
        {
            stdOutOfMemory();
        }
        ( (OptixInputState)gblState->optixInputState )->decryptedBufferLen = 0;
        ( (OptixInputState)gblState->optixInputState )->ptxInputLen        = 0;
        ( (OptixInputState)gblState->optixInputState )->lwrrentBufferSize  = 0;
        ( (OptixInputState)gblState->optixInputState )->decrypter          = decrypter;
        ( (OptixInputState)gblState->optixInputState )->decryptCall        = decryptionCB;
    }

    ptxPushSourcePos(gblState, newFileName, newLineNo);
}

void ptxInitScanner(ptxParsingState gblState)
{
    gblState->parseData->pushedInputs       = NULL; 
    gblState->parseData->pushedPending      = NULL; 
    gblState->parseData->pushedInput        = NULL;             
    gblState->parseData->pushedSourcePos    = NULL; 
    gblState->parseData->pushedObfuscators  = NULL; 
    gblState->parseData->ptxLookahead       = 0;
    gblState->parseData->ptxCount           = 0;
    // OPTIX_HAND_EDIT
    gblState->optixInputState = NULL;
}

/*------------------------------ Definitions ---------------------------------*/

static String skip_until(ptxParsingState gblState, String str, Bool returnSkipped, int (*input)(), Bool allowEOF)
{
#define wrap_around_inc(i, lim) ((i) + 1 == (lim) ? 0 : (i) + 1)

        stdString_t result= NULL;
        Char  c;
        Int  i = 0, j, k, len = strlen(str);
        Int  start_mdline = ptxget_lineno(gblState->scanner);
        Char buf[1024];
        stdMEMCLEAR_S(buf, len);
 
        if (returnSkipped) { result= stringNEW(); }
 
        while ((c=input(gblState->scanner)) != 0) {
                if (c == EOF) {
                    if (allowEOF) { goto end; }
                    break;
                }
                if (returnSkipped) { stringAddChar(result,c); }
                
                buf[i] = c;
                i = wrap_around_inc(i, len);
 
                for(j = 0, k = i; j < len; j++, k = wrap_around_inc(k, len))
                {
                        if(str[j] != buf[k])
                                break;
                }
 
                if(j == len) { goto end; }
        }
 
        ptxset_lineno(start_mdline, gblState->scanner);
        ptxerror(gblState->scanner, gblState, "unexpected EOF while scanning");
        
    end:
        if (returnSkipped) {
            String res  = stringStripToBuf(result);
            uInt   len  = strlen(res);
            uInt   tlen = strlen(str);

            if (len >= tlen) {
                res[len-tlen]= 0;
            }

            return res;
        } else {
            return NULL;
        }
}


/*
 * special case string to integer colwerter,
 * for strings that can contain hex or decimal 
 * or octal or binary characters, and do not 
 * exceed 64 bits of integer representation:
 */
static void atox( Char *s, int base, uInt64 *result, Bool *isSigned )
{
    *result   = 0;
    *isSigned = True;

    while (*s) {
        Char c= *(s++);

        if (c == 'U')         { *isSigned = False;  return; }

        // set errno on overflow and set value to MAX UNSIGNED INT
        if (*result & U64_CONST(0x8000000000000000)) {
            errno   = ERANGE;
            *result = U64_CONST(0xFFFFFFFFFFFFFFFF);
            return;
        }
        *result *= base;

        if ('0'<=c && c<='9') { *result +=    c-'0'; }
        if ('a'<=c && c<='f') { *result += 10+c-'a'; }
        if ('A'<=c && c<='F') { *result += 10+c-'A'; } 

        // unsigned if value would overflow signed representation
        if (*result & U64_CONST(0x8000000000000000)) {
            *isSigned = False;
        }
    }
}


/*------------------------------ Definitions ---------------------------------*/

#define matchBuf(buf,i,s) matchBuf1(buf,i,s,sizeof(s)-1) 

static Bool matchBuf1(String buf, Int i, String s, Int size)
{
    return (  i >= size
           && !strncmp(&buf[i-size],s,size)
            );
}


static Int bufferNext( String buff, uInt i, uInt size, stdString_t s, Int c)
{
    if (i == size) {
        Int  slack   = 32;
        Int  saveTop = size-slack;
        Char saved   = buff[saveTop];
        buff[saveTop]= 0;
        if (s) { stringAddBuf(s,buff); }
        buff[saveTop]= saved;
        memcpy(&buff[0],&buff[saveTop],slack);
        i=slack;
    }
    
    buff[i++]= c;
    return i;
}


static uInt readComment(ptxParsingState gblState,  String buff, uInt i, uInt size, stdString_t s, String terminator )
{
    uInt terminatorLen= strlen(terminator);

    while (True) {
        Char c;
        if (!my_input(gblState, &c,False,True)) { ptxerror(gblState->scanner, gblState, "unexpected EOF while scanning"); }
        i= bufferNext(buff,i,size,s,c);
        
        if (matchBuf1(buff,i,terminator,terminatorLen)) {
            return i;
        }
    }
}



static uInt readENDMACRO( ptxParsingState gblState, String buff, uInt i, uInt size, stdString_t s, Bool addEnd )
{
    while (True) {
        Char c;
        my_input(gblState, &c,False,True);
        stdCHECK_WITH_POS( c, (ptxMsgMissingENDMACRO, ptxsLwrPos(gblState)) );
        i= bufferNext(buff,i,size,s,c);
        
        if (matchBuf(buff,i,".ENDMACRO")) {
            if (!addEnd) { i-= 9; }
            return i;
        } else 
        if (matchBuf(buff,i,"//")) {
            i= readComment(gblState, buff,i,size,s,"\n");
        } else 
        if (matchBuf(buff,i,"/*")) {
            i= readComment(gblState, buff,i,size,s,"*/");
        }
    }
}


String ptxReadMacroBody(ptxParsingState gblState)
{
    Char buff[1024];
    Int  i;
    stdString_t macro= stringNEW();
    
    i= readENDMACRO( gblState, buff, 0, sizeof(buff), macro, False );
    buff[i]= 0; stringAddBuf(macro,buff);
    
    return stringStripToBuf(macro);
}

int unrecognizedString(String *stringValPtr, String ptxtext, int ptxleng)
{
    /*  .quad, .byte and .4byte which needs recognizing only around debug
     *  directives, use this utility in non-debug context to pass them
     *  un-recognized by lexer
     */
    *stringValPtr = stdCOPYSTRING(ptxtext + 1); // to skip '.'
    (*stringValPtr)[ptxleng - 2] = '\0';
    return T_STRING;
}

%}

/* -------------------------------------------------------------------- */

%option noyywrap
%option yylineno
%option reentrant
%option bison-bridge

ws                      [ \t\015\032\014]
letter                  [a-zA-Z]
bit                     [0-1]
digit                   [0-9]
nonzerodigit            [1-9]
octdigit                [0-7]
hexdigit                [0-9A-Fa-f]
newline                 \n
followsym               ({letter}|{digit}|[_$])
id                      ([_$%]{followsym}+)|({letter}{followsym}*)
dotid                   (\.{letter}{followsym}*)
string                  \"[^\"]*\"
sourcePos               \#{ws}*(line{ws}*)?{digit}+{ws}+{string}
exponent                [Ee][+-]?{digit}+

hilo                    ("mul"|"mul24"|"mad"|"madc"|"mad24"|"clmad"|"dp2a")(".hi"|".lo")
fusedhilo               ("mad"|"madc")".fused"(".hi"|".lo")
wide                    ("mul"|"mad")".wide"
full                    ("div")".full"
surf                    ("suld"|"sust"|"_sulea"|"sured")(".b"|".p")
surf2                   ("_suld.b"|"_sust.b")
shf                     ("shf")(".l"|".r")
tex                     ("tex")(".base"|".level"|".grad"|".clamp"|".grad.clamp")
txq                     ("txq")(".level")
bar                     ("bar")(".red"|".scan"|".warp"|".arrive")
barrier                 ("barrier")(".red"|".scan"|".warp"|".arrive"|".cluster"|".cluster.arrive"|".cluster.wait")
mbarrier1               ("mbarrier")(".init"|".arrive"|".arrive_drop"|".test_wait"|".test_wait.parity"|".try_wait"|".try_wait.parity")
mbarrier2               ("mbarrier")(".pending_count"|".ilwal"|".expect_copy"|".arrive.expect_copy"|".arrive_drop.expect_copy"|".expect_tx"|".arrive.expect_tx"|".arrive_drop.expect_tx")
cp                      ("cp.async")(""|".mbarrier.arrive"|".wait_all"|".wait_group"|".commit_group")
setmaxreg               ("setmaxreg")(".try_alloc"|".alloc"|".dealloc"|".release")
setsmemsize             ("setsmemsize")(""|".flush")
setctaid                ("setctaid")(".x"|".y"|".z")
bulkcopy                ("cp.async.bulk")
bulkreduce              ("cp.reduce.async.bulk")
tensorcopy              ({bulkcopy})(".tensor")
tensorreduce            ({bulkreduce})(".tensor")
wmma                    "wmma."("mma"|"load.a"|"load.b"|"load.c"|"store.d")
misc                    ("ld.bv"|"cvta.to"|"cvt.pack"|"_checkfp.divide"|"pmevent.mask"|"intr.cpu"|"brx.idx"|"elect.one")
warpgroup               "_mma"(".warpgroup")
arrivewait              "_warpgroup"(".arrive"|".wait"|".commit_batch")
createpolicy            ("createpolicy")(".fractional"|".range"|".cvt")
fence                   ("fence")(""|".proxy")
membar                  ("membar")(""|".proxy")
xxid1                   ({id}|{hilo}|{fusedhilo}|{wide}|{full}|{surf}|{surf2}|{shf}|{tex}|{txq}|{bar}|{barrier}|{createpolicy})
xxid2                   ({mbarrier1}|{mbarrier2}|{misc}|{setctaid}|{wmma}|{warpgroup}|{arrivewait}|{setmaxreg}|{setsmemsize})
xxid3                   ({cp}|{tensorcopy}|{bulkcopy}|{bulkreduce}|{tensorreduce}|{fence}|{membar})
xxid                    ({xxid1}|{xxid2}|{xxid3})

%o 300000
%a 200000
%p 150000
%n 150000
%e 150000

%%

".DEFINE"{ws}+{id}      { 
                            uInt len;
                            String v = yytext+sizeof(".DEFINE");
                            String d = skip_until(gblState, "\n",True,input,False);
                            
                            while (stdIsSpace(*v,False)) { v++; }
                            while (stdIsSpace(*d,False)) { d++; }
                            
                            len= strlen(d);
                            while (len && d[len-1]=='\r') { 
                               d[--len]=0;
                            }
                            
                            ptxDefinePreprocessorMacro(gblState, stdCOPYSTRING(v),d);
                        }
".MACRO"                { return T_MACRO;     }

".version"{ws}+{digit}+\.{digit}+ {
                            String v=yytext+sizeof(".version");
                            while (stdIsSpace(*v,False)) { v++; }
                            yylval->stringVal = stdCOPYSTRING(v);
                            return T_VERSION;
                        }

".target"               { return T_TARGET;    }
".address_size"         { return T_ADDRSIZE;  }

".section"              { return T_SECTION;   }
".file"                 { return T_FILE;      }
".loc"                  { return T_LOC;       }
"inlined_at"            { return T_INLINED_AT;}
"function_name"         { return T_FUNCNAME;  }

".metadata_section"     { return T_METADATA_SECTION; }
".metadata_index"       { return T_METADATA_INDEX;   }
".metadata"             { return T_METADATA;         }

".pragma"               { return T_PRAGMA;    }

".alias"                { return T_ALIAS;     }

".align"                { return T_ALIGN;     }
".allocno"              { return T_ALLOCNO;   }
".retaddr_allocno"      { return T_RETADDR_ALLOCNO; }
".scratch"              { return T_SCRATCH;   }
".func"                 { return T_FUNCTION;  }
".entry"                { return T_ENTRY;     }
".maxnreg"              { return T_MAXNREG;   }
".maxntid"              { return T_MAXNTID;   }
".maxnctapersm"         { return T_MAXNCTAPERSM; }
".minnctapersm"         { return T_MINNCTAPERSM; }
".reqntid"              { return T_REQNTID;   }
".noreturn"             { return T_NORETURN;  }
".unique"               { return T_UNIQUE;    }

".extern"               { return T_EXTERN;    }
".visible"              { return T_VISIBLE;   }
".weak"                 { return T_WEAK;      }
".common"               { return T_COMMON;    }
".FORCE_INLINE"         { return T_FORCEINLINE;    }

".ptr"                  { return T_POINTER;   }

".attribute"            { return T_ATTRIBUTE; }
".managed"              { return T_MANAGED;   } 

".branchtargets"        { return T_BRANCHTARGETS;   }
".calltargets"          { return T_CALLTARGETS;     }
".callprototype"        { return T_CALLPROTOTYPE;   }

"WARP_SZ"               { return T_WARP_SZ;   }

"<<"                    { return T_SHL;       }
">>"                    { return T_SHR;       }
"=="                    { return T_EQ;        }
"!="                    { return T_NEQ;       }
"<="                    { return T_LTEQ;      }
">="                    { return T_GTEQ;      }
"||"                    { return T_OROR;      }
"&&"                    { return T_ANDAND;    }

".reg"                  { return T_REG;       }
".const"                { yylval->intVal = 0xffffffff;          return T_CONST; }
".const["{digit}"]"     { yylval->intVal =    *(yytext+7)-'0';  return T_CONST; }
".const[1"{digit}"]"    { yylval->intVal = 10+*(yytext+8)-'0';  return T_CONST; }
".global"               { return T_GLOBAL;    }
".local"                { return T_LOCAL;     }
".param"                { return T_PARAM;     }
".shared"               { return T_SHARED;    }

".eq"                   { yylval->cmpVal = ptxEQ_Comparison;  return T_COMPARISON; }
".ne"                   { yylval->cmpVal = ptxNE_Comparison;  return T_COMPARISON; }
".lt"                   { yylval->cmpVal = ptxLT_Comparison;  return T_COMPARISON; }
".le"                   { yylval->cmpVal = ptxLE_Comparison;  return T_COMPARISON; }
".gt"                   { yylval->cmpVal = ptxGT_Comparison;  return T_COMPARISON; }
".ge"                   { yylval->cmpVal = ptxGE_Comparison;  return T_COMPARISON; }
".lo"                   { yylval->cmpVal = ptxLO_Comparison;  return T_COMPARISON; }
".ls"                   { yylval->cmpVal = ptxLS_Comparison;  return T_COMPARISON; }
".hi"                   { yylval->cmpVal = ptxHI_Comparison;  return T_COMPARISON; }
".hs"                   { yylval->cmpVal = ptxHS_Comparison;  return T_COMPARISON; }
".num"                  { yylval->cmpVal = ptxNUM_Comparison; return T_COMPARISON; }
".nan"                  { yylval->cmpVal = ptxNAN_Comparison; return T_COMPARISON; }
".equ"                  { yylval->cmpVal = ptxEQU_Comparison; return T_COMPARISON; }
".neu"                  { yylval->cmpVal = ptxNEU_Comparison; return T_COMPARISON; }
".ltu"                  { yylval->cmpVal = ptxLTU_Comparison; return T_COMPARISON; }
".leu"                  { yylval->cmpVal = ptxLEU_Comparison; return T_COMPARISON; }
".gtu"                  { yylval->cmpVal = ptxGTU_Comparison; return T_COMPARISON; }
".geu"                  { yylval->cmpVal = ptxGEU_Comparison; return T_COMPARISON; }

".and"                  { yylval->opVal = ptxANDOp;       return T_BOOLEAN_OP; }
".or"                   { yylval->opVal = ptxOROp;        return T_BOOLEAN_OP; }
".xor"                  { yylval->opVal = ptxXOROp;       return T_BOOLEAN_OP; }

".cas"                  { yylval->opVal = ptxCASOp;       return T_ATOMIC_OP;  }
".exch"                 { yylval->opVal = ptxEXCHOp;      return T_ATOMIC_OP;  }
".inc"                  { yylval->opVal = ptxINCOp;       return T_ATOMIC_OP;  }
".dec"                  { yylval->opVal = ptxDECOp;       return T_ATOMIC_OP;  }
".safeadd"              { yylval->opVal = ptxSAFEADDOp;   return T_ATOMIC_OP;  }

".noinc"                { yylval->opVal = ptxNOINC_MOD;     return T_NOINCMOD;   }

".add"                  { yylval->opVal = ptxADDOp;       return T_ARITH_OP;   }
".min"                  { yylval->opVal = ptxMINOp;       return T_ARITH_OP;   }
".max"                  { yylval->opVal = ptxMAXOp;       return T_ARITH_OP;   }
".maxabs"               { yylval->opVal = ptxMAXABSOp;    return T_ARITH_OP;   }
".popc"                 { yylval->opVal = ptxPOPCOp;      return T_ARITH_OP;   }  /* bar.red only */

".approx"               { yylval->intVal = ptxAPRX_MOD;                         return T_APRXMOD;  }
".relu"                 { yylval->intVal = ptxRELU_MOD;                         return T_RELUMOD;  }
".ftz"                  { yylval->intVal = ptxFTZ_MOD;                          return T_FTZMOD;   }
".noftz"                { yylval->intVal = ptxEXPLICIT_NOFTZ_MOD;               return T_NOFTZMOD; }
".sat"                  { yylval->intVal = ptxSAT_MOD;                          return T_SATMOD;   }
".satfinite"            { yylval->intVal = ptxSATFINITE_MOD;                    return T_SATFMOD;  }
".cc"                   { yylval->intVal = ptxCC_MOD;                           return T_CCMOD;    }
".shiftamt"             { yylval->intVal = ptxSHAMT_MOD;                        return T_SHAMTMOD; }
".NaN"                  { yylval->intVal = ptxNANMODE_MOD;                      return T_NANMODEMOD;  }
".mbarrier"             { yylval->intVal = ptxMBARRIER_MOD;                     return T_MBARRIERMOD; }
".sync"                 { yylval->intVal = ptxSYNC_MOD;                         return T_SYNCMOD;  }

".noComplete"           { yylval->intVal = ptxNOCOMPLETE_MOD;                   return T_NOCOMPLETEMOD;  }

".cta"                  { yylval->intVal = ptxCTA_MOD;                          return T_SCOPEMOD; }
".gl"                   { yylval->intVal = ptxGL_MOD;                           return T_SCOPEMOD; }
".gpu"                  { yylval->intVal = ptxGPU_MOD;                          return T_SCOPEMOD; }
".sys"                  { yylval->intVal = ptxSYS_MOD;                          return T_SCOPEMOD; }

".volatile"             { yylval->intVal = ptxVOLATILE_MOD;                     return T_ORDERMOD; }
".relaxed"              { yylval->intVal = ptxRELAXED_MOD;                      return T_ORDERMOD; }
".acquire"              { yylval->intVal = ptxACQUIRE_MOD;                      return T_ORDERMOD; }
".release"              { yylval->intVal = ptxRELEASE_MOD;                      return T_ORDERMOD; }
".acq_rel"              { yylval->intVal = ptxACQREL_MOD;                       return T_ORDERMOD; }
".sc"                   { yylval->intVal = ptxSC_MOD;                           return T_ORDERMOD; }

".nc"                   { yylval->intVal = ptxNC_MOD;                           return T_NCMOD;    }

".finite"               { yylval->intVal = ptxFINITE_MOD;                       return T_TESTPMOD; }
".infinite"             { yylval->intVal = ptxINFINITE_MOD;                     return T_TESTPMOD; }
".number"               { yylval->intVal = ptxNUMBER_MOD;                       return T_TESTPMOD; }
".notanumber"           { yylval->intVal = ptxNOTANUMBER_MOD;                   return T_TESTPMOD; }
".normal"               { yylval->intVal = ptxNORMAL_MOD;                       return T_TESTPMOD; }
".subnormal"            { yylval->intVal = ptxSUBNORMAL_MOD;                    return T_TESTPMOD; }

".ca"                   { yylval->intVal = ptxCA_MOD;                           return T_COPMOD;   }
".cg"                   { yylval->intVal = ptxCG_MOD;                           return T_COPMOD;   }
".cs"                   { yylval->intVal = ptxCS_MOD;                           return T_COPMOD;   }
".lu"                   { yylval->intVal = ptxLU_MOD;                           return T_COPMOD;   }
".cv"                   { yylval->intVal = ptxCV_MOD;                           return T_COPMOD;   }
".wb"                   { yylval->intVal = ptxWB_MOD;                           return T_COPMOD;   }
".wt"                   { yylval->intVal = ptxWT_MOD;                           return T_COPMOD;   }
".ilw"                  { yylval->intVal = ptxILW_MOD;                          return T_COPMOD;   }
".ilwall"               { yylval->intVal = ptxILWALL_MOD;                       return T_COPMOD;   }

".L1"                   { yylval->intVal = ptxL1_MOD;                           return T_LVLMOD;   }
".L2"                   { yylval->intVal = ptxL2_MOD;                           return T_LVLMOD;   }

".rn"                   { yylval->intVal = ptxRN_MOD;                           return T_ROUNDMOD; }
".rna"                  { yylval->intVal = ptxRNA_MOD;                          return T_ROUNDMOD; }
".rm"                   { yylval->intVal = ptxRM_MOD;                           return T_ROUNDMOD; }
".rp"                   { yylval->intVal = ptxRP_MOD;                           return T_ROUNDMOD; }
".rz"                   { yylval->intVal = ptxRZ_MOD;                           return T_ROUNDMOD; }
".rni"                  { yylval->intVal = ptxRNI_MOD;                          return T_ROUNDMOD; }
".rmi"                  { yylval->intVal = ptxRMI_MOD;                          return T_ROUNDMOD; }
".rpi"                  { yylval->intVal = ptxRPI_MOD;                          return T_ROUNDMOD; }
".rzi"                  { yylval->intVal = ptxRZI_MOD;                          return T_ROUNDMOD; }

".v2"                   { yylval->intVal = ptxV2_MOD;                           return T_VECTOR;   }
".v4"                   { yylval->intVal = ptxV4_MOD;                           return T_VECTOR;   }

 /* Ideantify dimensionality modifier and specialize to texmod or tensorDim later. */
".1d"                   { yylval->intVal = ptxDIM1D_MOD;                        return T_DIMMOD;   }
".2d"                   { yylval->intVal = ptxDIM2D_MOD;                        return T_DIMMOD;   }
".3d"                   { yylval->intVal = ptxDIM3D_MOD;                        return T_DIMMOD;   }
".4d"                   { yylval->intVal = ptxDIM4D_MOD;                        return T_DIMMOD;   }
".5d"                   { yylval->intVal = ptxDIM5D_MOD;                        return T_DIMMOD;   }

".1d_buffer"            { yylval->intVal = ptx1D_BUFFER_MOD;                    return T_TEXMOD;   }
".a1d"                  { yylval->intVal = ptxA1D_MOD;                          return T_TEXMOD;   }
".a2d"                  { yylval->intVal = ptxA2D_MOD;                          return T_TEXMOD;   }
".lwbe"                 { yylval->intVal = ptxLWBE_MOD;                         return T_TEXMOD;   }
".alwbe"                { yylval->intVal = ptxALWBE_MOD;                        return T_TEXMOD;   }
".2dms"                 { yylval->intVal = ptx2DMS_MOD;                         return T_TEXMOD;   }
".a2dms"                { yylval->intVal = ptxA2DMS_MOD;                        return T_TEXMOD;   }
".ballot"               { yylval->intVal = ptxBALLOT_MOD;                       return T_VOTEMOD;  }

".footprint"            { yylval->intVal = ptxFOOTPRINT_MOD;                    return T_FOOTPRINTMOD; }
".coarse"               { yylval->intVal = ptxCOARSE_MOD;                       return T_COARSEMOD; }

 /* .any/.all are used by multiple instructions, so a generic modifier is first created, then */
 /* turned into the correct modifier once the instruction has been recognized.                */
".all"                  { yylval->intVal = ptxTALL_MOD;                         return T_THREADSMOD; }
".any"                  { yylval->intVal = ptxTANY_MOD;                         return T_THREADSMOD; }

".aligned"              { yylval->intVal = ptxALIGNED_MOD;                      return T_ALIGNMOD;   }

 /* .uni is used by multiple instructions, so a generic modifier is first created, then      */
 /* turned into the correct modifier once the instruction has been recognized.               */
".uni"                  { yylval->intVal = ptxUNIFORM_MOD;                      return T_UNIFORMMOD; }
".unanimous"            { yylval->intVal = ptxUNANI_MOD;                        return T_BRANCHMOD;  }
".colw"                 { yylval->intVal = ptxCOLW_MOD;                         return T_BRANCHMOD;  }
".div"                  { yylval->intVal = ptxDIV_MOD;                          return T_BRANCHMOD;  }

".exclusive"            { yylval->intVal = ptxEXCLUSIVE_MOD;                    return T_EXCLUSIVE;  }

 /* modifiers for video/surface instruction support */
".clamp"                { yylval->intVal = ptxCLAMP_MOD;                        return T_CLAMPMOD; }
".wrap"                 { yylval->intVal = ptxWRAP_MOD;                         return T_CLAMPMOD; }
".trap"                 { yylval->intVal = ptxTRAP_MOD;                         return T_CLAMPMOD; }
".zero"                 { yylval->intVal = ptxZERO_MOD;                         return T_CLAMPMOD; }

".shr7"                 { yylval->intVal = ptxSHR7_MOD;                         return T_SHRMOD;   }
".shr15"                { yylval->intVal = ptxSHR15_MOD;                        return T_SHRMOD;   }

".dual"                 { yylval->intVal = ptxTTUSLOT_DUAL_MOD;                 return T_TTUSLOTMOD; }

".close"                { yylval->intVal = ptxTTU_CLOSE_MOD;                    return T_TTUMOD;   }

".trans"                { yylval->intVal = ptxTRANS_MOD;                        return T_TRANSMOD; }
".seq"                  { yylval->intVal = ptxSEQ_MOD;                          return T_SEQMOD;   }
".expand"               { yylval->intVal = ptxEXPAND_MOD;                       return T_EXPANDMOD; }

".x1"                   { yylval->intVal = ptxNUM_x1_MOD;                       return T_NUMMOD;   }
".x2"                   { yylval->intVal = ptxNUM_x2_MOD;                       return T_NUMMOD;   }
".x4"                   { yylval->intVal = ptxNUM_x4_MOD;                       return T_NUMMOD;   }

".1g"                   { yylval->intVal = ptxGROUP_g1_MOD;                     return T_GROUPMOD; }
".2g"                   { yylval->intVal = ptxGROUP_g2_MOD;                     return T_GROUPMOD; }
".4g"                   { yylval->intVal = ptxGROUP_g4_MOD;                     return T_GROUPMOD; }

 /* .abs modifier for min/max operations with .xorsign */
".abs"                  { yylval->intVal = ptxABS_MOD;                          return T_ABSMOD;   }

 /* .L2::cacheHint modifier for ld/st/atom/red/cp.async instructions for explicit descriptor */
".L2::cache_hint"       { yylval->intVal = ptxCACHEHINT_MOD;                    return T_CACHEHINTMOD; }

 /* .L2::evict_priority modifiers will be treated as level + eviction_priority in parser */
".L2::evict_first"      { yylval->intVal = ptxL2EVICTFIRST_MOD;                 return T_LEVELEVICTMOD;   }
".L2::evict_last"       { yylval->intVal = ptxL2EVICTLAST_MOD;                  return T_LEVELEVICTMOD;   }
".L2::evict_unchanged"  { yylval->intVal = ptxL2EVICTUNCHANGED_MOD;             return T_LEVELEVICTMOD;   }
".L2::evict_normal"     { yylval->intVal = ptxL2EVICTNORMAL_MOD;                return T_LEVELEVICTMOD;   }
".L2::no_allocate"      { yylval->intVal = ptxL2EVICTNOALLOC_MOD;               return T_LEVELEVICTMOD;   }
 /* .L1::evict_priority is syntax for eviction_priority for ld/st */
".L1::evict_first"      { yylval->intVal = ptxL1EVICTFIRST_MOD;                 return T_LEVELEVICTMOD;   }
".L1::evict_last"       { yylval->intVal = ptxL1EVICTLAST_MOD;                  return T_LEVELEVICTMOD;   }
".L1::evict_unchanged"  { yylval->intVal = ptxL1EVICTUNCHANGED_MOD;             return T_LEVELEVICTMOD;   }
".L1::evict_normal"     { yylval->intVal = ptxL1EVICTNORMAL_MOD;                return T_LEVELEVICTMOD;   }
".L1::no_allocate"      { yylval->intVal = ptxL1EVICTNOALLOC_MOD;               return T_LEVELEVICTMOD;   }

 /* NEG_AB and NEG_C modifiers are synthesized from optional operand negations */
".po"                   { yylval->intVal = ptxVMAD_PO_MOD;                      return T_VMADMOD;  }

 /* permute modifiers */
".f4e"                  { yylval->intVal = ptxPRMT_F4E_MOD;                     return T_PRMTMOD;  }
".b4e"                  { yylval->intVal = ptxPRMT_B4E_MOD;                     return T_PRMTMOD;  }
".rc8"                  { yylval->intVal = ptxPRMT_RC8_MOD;                     return T_PRMTMOD;  }
".ecl"                  { yylval->intVal = ptxPRMT_ECL_MOD;                     return T_PRMTMOD;  }
".ecr"                  { yylval->intVal = ptxPRMT_ECR_MOD;                     return T_PRMTMOD;  }
".rc16"                 { yylval->intVal = ptxPRMT_RC16_MOD;                    return T_PRMTMOD;  }

 /* shuffle modifiers */
".up"                   { yylval->intVal = ptxSHFL_UP_MOD;                      return T_SHFLMOD;  }
".down"                 { yylval->intVal = ptxSHFL_DOWN_MOD;                    return T_SHFLMOD;  }
".bfly"                 { yylval->intVal = ptxSHFL_BFLY_MOD;                    return T_SHFLMOD;  }
".idx"                  { yylval->intVal = ptxSHFL_IDX_MOD;                     return T_SHFLMOD;  }

 /* endis modifiers */
".en"                   { yylval->intVal = ptxENDIS_EN_MOD;                     return T_ENDISMOD; }
".dis"                  { yylval->intVal = ptxENDIS_DIS_MOD;                    return T_ENDISMOD; }

 /* rand modifier */
".rand"                 { yylval->intVal = ptxRAND_MOD;                         return T_RANDMOD;  }

".s2"                   { yylval->intVal = ptxTYPE_s2_MOD;                      return T_TYPEMOD;  }
".s4"                   { yylval->intVal = ptxTYPE_s4_MOD;                      return T_TYPEMOD;  }
".s8"                   { yylval->typeVal= ptxCreateIntType(8,True, gblState);            return T_TYPENAME; }
".s16"                  { yylval->typeVal= ptxCreateIntType(16,True, gblState);           return T_TYPENAME; }
".s32"                  { yylval->typeVal= ptxCreateIntType(32,True, gblState);           return T_TYPENAME; }
".s64"                  { yylval->typeVal= ptxCreateIntType(64,True, gblState);           return T_TYPENAME; }

".u2"                   { yylval->intVal = ptxTYPE_u2_MOD;                      return T_TYPEMOD;  }
".u4"                   { yylval->intVal = ptxTYPE_u4_MOD;                      return T_TYPEMOD;  }
".u8"                   { yylval->typeVal= ptxCreateIntType(8,False, gblState);           return T_TYPENAME; }
".u16"                  { yylval->typeVal= ptxCreateIntType(16,False, gblState);          return T_TYPENAME; }
".u32"                  { yylval->typeVal= ptxCreateIntType(32,False, gblState);          return T_TYPENAME; }
".u64"                  { yylval->typeVal= ptxCreateIntType(64,False, gblState);          return T_TYPENAME; }

".e4m3"                 { yylval->typeVal= ptxCreateLwstomFloatType(4, 3, 1, gblState);   return T_TYPENAME; }
".e5m2"                 { yylval->typeVal= ptxCreateLwstomFloatType(5, 2, 1, gblState);   return T_TYPENAME; }
".e4m3x2"               { yylval->typeVal= ptxCreateLwstomFloatType(4, 3, 2, gblState);   return T_TYPENAME; }
".e5m2x2"               { yylval->typeVal= ptxCreateLwstomFloatType(5, 2, 2, gblState);   return T_TYPENAME; }
".f16"                  { yylval->typeVal= ptxCreateFloatType(16, gblState);              return T_TYPENAME; }
".f16x2"                { yylval->typeVal= ptxCreatePackedHalfFloatType(32, gblState);    return T_TYPENAME; }
".f32"                  { yylval->typeVal= ptxCreateFloatType(32, gblState);              return T_TYPENAME; }
".f64"                  { yylval->typeVal= ptxCreateFloatType(64, gblState);              return T_TYPENAME; }

".b8"                   { yylval->typeVal= ptxCreateBitType(8, gblState);                 return T_TYPENAME; }
".b16"                  { yylval->typeVal= ptxCreateBitType(16, gblState);                return T_TYPENAME; }
".b32"                  { yylval->typeVal= ptxCreateBitType(32, gblState);                return T_TYPENAME; }
".b64"                  { yylval->typeVal= ptxCreateBitType(64, gblState);                return T_TYPENAME; }
".b128"                 { yylval->typeVal= ptxCreateBitType(128, gblState);               return T_TYPENAME; }

".byte"                 { if (gblState->parseData->parsingDwarfData) {
                              yylval->typeVal= ptxCreateBitType(8, gblState);             return T_TYPENAME;
                          }
                          return unrecognizedString(&yylval->stringVal, yytext, yyleng);
                        }
".4byte"                { if (gblState->parseData->parsingDwarfData) {
                              yylval->typeVal= ptxCreateBitType(32, gblState);            return T_TYPENAME;
                          }
                          return unrecognizedString(&yylval->stringVal, yytext, yyleng);
                        }

".thread"               { yylval->intVal = ptxTGTHREAD_MOD;                       return T_THREADGROUPMOD;  }
".pair"                 { yylval->intVal = ptxTGPAIR_MOD;                         return T_THREADGROUPMOD;  }
".quad"                 { if (gblState->parseData->parsingDwarfData) {
                              yylval->typeVal= ptxCreateBitType(64, gblState); return T_TYPENAME;
                          }
                          yylval->intVal = ptxTGQUAD_MOD;                       return T_THREADGROUPMOD;
                        }

".pred"                 { yylval->typeVal= ptxCreatePredicateType(gblState);            return T_TYPENAME; }

".m8n8k4"               { yylval->intVal = ptxSHAPE_080804_MOD;                 return T_SHAPE; }
".m8n8k16"              { yylval->intVal = ptxSHAPE_080816_MOD;                 return T_SHAPE; }
".m8n8k32"              { yylval->intVal = ptxSHAPE_080832_MOD;                 return T_SHAPE; }
".m16n16k8"             { yylval->intVal = ptxSHAPE_161608_MOD;                 return T_SHAPE; }
".m16n16k16"            { yylval->intVal = ptxSHAPE_161616_MOD;                 return T_SHAPE; }
".m32n8k16"             { yylval->intVal = ptxSHAPE_320816_MOD;                 return T_SHAPE; }
".m8n32k16"             { yylval->intVal = ptxSHAPE_083216_MOD;                 return T_SHAPE; }
".m16n8k8"              { yylval->intVal = ptxSHAPE_160808_MOD;                 return T_SHAPE; }
".m16n8k16"             { yylval->intVal = ptxSHAPE_160816_MOD;                 return T_SHAPE; }
".m16n8k32"             { yylval->intVal = ptxSHAPE_160832_MOD;                 return T_SHAPE; }
".m16n8k4"              { yylval->intVal = ptxSHAPE_160804_MOD;                 return T_SHAPE; }
".m16n8k64"             { yylval->intVal = ptxSHAPE_160864_MOD;                 return T_SHAPE; }
".m16n8k128"            { yylval->intVal = ptxSHAPE_1608128_MOD;                return T_SHAPE; }
".m16n8k256"            { yylval->intVal = ptxSHAPE_1608256_MOD;                return T_SHAPE; }
".m8n8k128"             { yylval->intVal = ptxSHAPE_0808128_MOD;                return T_SHAPE; }
".m8n8"                 { yylval->intVal = ptxSHAPE_080800_MOD;                 return T_SHAPE; }
".m8n16"                { yylval->intVal = ptxSHAPE_081600_MOD;                 return T_SHAPE; }
".m8n32"                { yylval->intVal = ptxSHAPE_083200_MOD;                 return T_SHAPE; }
".m8n64"                { yylval->intVal = ptxSHAPE_086400_MOD;                 return T_SHAPE; }
".m8n8k64"              { yylval->intVal = ptxSHAPE_080864_MOD;                 return T_SHAPE; }
".m64n8k32"             { yylval->intVal = ptxSHAPE_640832_MOD;                 return T_SHAPE; }
".m64n16k32"            { yylval->intVal = ptxSHAPE_641632_MOD;                 return T_SHAPE; }
".m64n24k32"            { yylval->intVal = ptxSHAPE_642432_MOD;                 return T_SHAPE; }
".m64n32k32"            { yylval->intVal = ptxSHAPE_643232_MOD;                 return T_SHAPE; }
".m64n40k32"            { yylval->intVal = ptxSHAPE_644032_MOD;                 return T_SHAPE; }
".m64n48k32"            { yylval->intVal = ptxSHAPE_644832_MOD;                 return T_SHAPE; }
".m64n56k32"            { yylval->intVal = ptxSHAPE_645632_MOD;                 return T_SHAPE; }
".m64n64k32"            { yylval->intVal = ptxSHAPE_646432_MOD;                 return T_SHAPE; }
".m64n72k32"            { yylval->intVal = ptxSHAPE_647232_MOD;                 return T_SHAPE; }
".m64n80k32"            { yylval->intVal = ptxSHAPE_648032_MOD;                 return T_SHAPE; }
".m64n88k32"            { yylval->intVal = ptxSHAPE_648832_MOD;                 return T_SHAPE; }
".m64n96k32"            { yylval->intVal = ptxSHAPE_649632_MOD;                 return T_SHAPE; }
".m64n104k32"           { yylval->intVal = ptxSHAPE_6410432_MOD;                return T_SHAPE; }
".m64n112k32"           { yylval->intVal = ptxSHAPE_6411232_MOD;                return T_SHAPE; }
".m64n120k32"           { yylval->intVal = ptxSHAPE_6412032_MOD;                return T_SHAPE; }
".m64n128k32"           { yylval->intVal = ptxSHAPE_6412832_MOD;                return T_SHAPE; }
".m64n136k32"           { yylval->intVal = ptxSHAPE_6413632_MOD;                return T_SHAPE; }
".m64n144k32"           { yylval->intVal = ptxSHAPE_6414432_MOD;                return T_SHAPE; }
".m64n152k32"           { yylval->intVal = ptxSHAPE_6415232_MOD;                return T_SHAPE; }
".m64n160k32"           { yylval->intVal = ptxSHAPE_6416032_MOD;                return T_SHAPE; }
".m64n168k32"           { yylval->intVal = ptxSHAPE_6416832_MOD;                return T_SHAPE; }
".m64n176k32"           { yylval->intVal = ptxSHAPE_6417632_MOD;                return T_SHAPE; }
".m64n184k32"           { yylval->intVal = ptxSHAPE_6418432_MOD;                return T_SHAPE; }
".m64n192k32"           { yylval->intVal = ptxSHAPE_6419232_MOD;                return T_SHAPE; }
".m64n200k32"           { yylval->intVal = ptxSHAPE_6420032_MOD;                return T_SHAPE; }
".m64n208k32"           { yylval->intVal = ptxSHAPE_6420832_MOD;                return T_SHAPE; }
".m64n216k32"           { yylval->intVal = ptxSHAPE_6421632_MOD;                return T_SHAPE; }
".m64n224k32"           { yylval->intVal = ptxSHAPE_6422432_MOD;                return T_SHAPE; }
".m64n232k32"           { yylval->intVal = ptxSHAPE_6423232_MOD;                return T_SHAPE; }
".m64n240k32"           { yylval->intVal = ptxSHAPE_6424032_MOD;                return T_SHAPE; }
".m64n248k32"           { yylval->intVal = ptxSHAPE_6424832_MOD;                return T_SHAPE; }
".m64n256k32"           { yylval->intVal = ptxSHAPE_6425632_MOD;                return T_SHAPE; }
".m64n8k16"             { yylval->intVal = ptxSHAPE_640816_MOD;                 return T_SHAPE; }
".m64n16k16"            { yylval->intVal = ptxSHAPE_641616_MOD;                 return T_SHAPE; }
".m64n24k16"            { yylval->intVal = ptxSHAPE_642416_MOD;                 return T_SHAPE; }
".m64n32k16"            { yylval->intVal = ptxSHAPE_643216_MOD;                 return T_SHAPE; }
".m64n40k16"            { yylval->intVal = ptxSHAPE_644016_MOD;                 return T_SHAPE; }
".m64n48k16"            { yylval->intVal = ptxSHAPE_644816_MOD;                 return T_SHAPE; }
".m64n56k16"            { yylval->intVal = ptxSHAPE_645616_MOD;                 return T_SHAPE; }
".m64n64k16"            { yylval->intVal = ptxSHAPE_646416_MOD;                 return T_SHAPE; }
".m64n72k16"            { yylval->intVal = ptxSHAPE_647216_MOD;                 return T_SHAPE; }
".m64n80k16"            { yylval->intVal = ptxSHAPE_648016_MOD;                 return T_SHAPE; }
".m64n88k16"            { yylval->intVal = ptxSHAPE_648816_MOD;                 return T_SHAPE; }
".m64n96k16"            { yylval->intVal = ptxSHAPE_649616_MOD;                 return T_SHAPE; }
".m64n104k16"           { yylval->intVal = ptxSHAPE_6410416_MOD;                return T_SHAPE; }
".m64n112k16"           { yylval->intVal = ptxSHAPE_6411216_MOD;                return T_SHAPE; }
".m64n120k16"           { yylval->intVal = ptxSHAPE_6412016_MOD;                return T_SHAPE; }
".m64n128k16"           { yylval->intVal = ptxSHAPE_6412816_MOD;                return T_SHAPE; }
".m64n136k16"           { yylval->intVal = ptxSHAPE_6413616_MOD;                return T_SHAPE; }
".m64n144k16"           { yylval->intVal = ptxSHAPE_6414416_MOD;                return T_SHAPE; }
".m64n152k16"           { yylval->intVal = ptxSHAPE_6415216_MOD;                return T_SHAPE; }
".m64n160k16"           { yylval->intVal = ptxSHAPE_6416016_MOD;                return T_SHAPE; }
".m64n168k16"           { yylval->intVal = ptxSHAPE_6416816_MOD;                return T_SHAPE; }
".m64n176k16"           { yylval->intVal = ptxSHAPE_6417616_MOD;                return T_SHAPE; }
".m64n184k16"           { yylval->intVal = ptxSHAPE_6418416_MOD;                return T_SHAPE; }
".m64n192k16"           { yylval->intVal = ptxSHAPE_6419216_MOD;                return T_SHAPE; }
".m64n200k16"           { yylval->intVal = ptxSHAPE_6420016_MOD;                return T_SHAPE; }
".m64n208k16"           { yylval->intVal = ptxSHAPE_6420816_MOD;                return T_SHAPE; }
".m64n216k16"           { yylval->intVal = ptxSHAPE_6421616_MOD;                return T_SHAPE; }
".m64n224k16"           { yylval->intVal = ptxSHAPE_6422416_MOD;                return T_SHAPE; }
".m64n232k16"           { yylval->intVal = ptxSHAPE_6423216_MOD;                return T_SHAPE; }
".m64n240k16"           { yylval->intVal = ptxSHAPE_6424016_MOD;                return T_SHAPE; }
".m64n248k16"           { yylval->intVal = ptxSHAPE_6424816_MOD;                return T_SHAPE; }
".m64n256k16"           { yylval->intVal = ptxSHAPE_6425616_MOD;                return T_SHAPE; }
".m64n8k8"              { yylval->intVal = ptxSHAPE_640808_MOD;                 return T_SHAPE; }
".m64n16k8"             { yylval->intVal = ptxSHAPE_641608_MOD;                 return T_SHAPE; }
".m64n24k8"             { yylval->intVal = ptxSHAPE_642408_MOD;                 return T_SHAPE; }
".m64n32k8"             { yylval->intVal = ptxSHAPE_643208_MOD;                 return T_SHAPE; }
".m64n40k8"             { yylval->intVal = ptxSHAPE_644008_MOD;                 return T_SHAPE; }
".m64n48k8"             { yylval->intVal = ptxSHAPE_644808_MOD;                 return T_SHAPE; }
".m64n56k8"             { yylval->intVal = ptxSHAPE_645608_MOD;                 return T_SHAPE; }
".m64n64k8"             { yylval->intVal = ptxSHAPE_646408_MOD;                 return T_SHAPE; }
".m64n72k8"             { yylval->intVal = ptxSHAPE_647208_MOD;                 return T_SHAPE; }
".m64n80k8"             { yylval->intVal = ptxSHAPE_648008_MOD;                 return T_SHAPE; }
".m64n88k8"             { yylval->intVal = ptxSHAPE_648808_MOD;                 return T_SHAPE; }
".m64n96k8"             { yylval->intVal = ptxSHAPE_649608_MOD;                 return T_SHAPE; }
".m64n104k8"            { yylval->intVal = ptxSHAPE_6410408_MOD;                return T_SHAPE; }
".m64n112k8"            { yylval->intVal = ptxSHAPE_6411208_MOD;                return T_SHAPE; }
".m64n120k8"            { yylval->intVal = ptxSHAPE_6412008_MOD;                return T_SHAPE; }
".m64n128k8"            { yylval->intVal = ptxSHAPE_6412808_MOD;                return T_SHAPE; }
".m64n136k8"            { yylval->intVal = ptxSHAPE_6413608_MOD;                return T_SHAPE; }
".m64n144k8"            { yylval->intVal = ptxSHAPE_6414408_MOD;                return T_SHAPE; }
".m64n152k8"            { yylval->intVal = ptxSHAPE_6415208_MOD;                return T_SHAPE; }
".m64n160k8"            { yylval->intVal = ptxSHAPE_6416008_MOD;                return T_SHAPE; }
".m64n168k8"            { yylval->intVal = ptxSHAPE_6416808_MOD;                return T_SHAPE; }
".m64n176k8"            { yylval->intVal = ptxSHAPE_6417608_MOD;                return T_SHAPE; }
".m64n184k8"            { yylval->intVal = ptxSHAPE_6418408_MOD;                return T_SHAPE; }
".m64n192k8"            { yylval->intVal = ptxSHAPE_6419208_MOD;                return T_SHAPE; }
".m64n200k8"            { yylval->intVal = ptxSHAPE_6420008_MOD;                return T_SHAPE; }
".m64n208k8"            { yylval->intVal = ptxSHAPE_6420808_MOD;                return T_SHAPE; }
".m64n216k8"            { yylval->intVal = ptxSHAPE_6421608_MOD;                return T_SHAPE; }
".m64n224k8"            { yylval->intVal = ptxSHAPE_6422408_MOD;                return T_SHAPE; }
".m64n232k8"            { yylval->intVal = ptxSHAPE_6423208_MOD;                return T_SHAPE; }
".m64n240k8"            { yylval->intVal = ptxSHAPE_6424008_MOD;                return T_SHAPE; }
".m64n248k8"            { yylval->intVal = ptxSHAPE_6424808_MOD;                return T_SHAPE; }
".m64n256k8"            { yylval->intVal = ptxSHAPE_6425608_MOD;                return T_SHAPE; }
".m64n8k64"             { yylval->intVal = ptxSHAPE_640864_MOD;                 return T_SHAPE; }
".m64n16k64"            { yylval->intVal = ptxSHAPE_641664_MOD;                 return T_SHAPE; }
".m64n24k64"            { yylval->intVal = ptxSHAPE_642464_MOD;                 return T_SHAPE; }
".m64n32k64"            { yylval->intVal = ptxSHAPE_643264_MOD;                 return T_SHAPE; }
".m64n40k64"            { yylval->intVal = ptxSHAPE_644064_MOD;                 return T_SHAPE; }
".m64n48k64"            { yylval->intVal = ptxSHAPE_644864_MOD;                 return T_SHAPE; }
".m64n56k64"            { yylval->intVal = ptxSHAPE_645664_MOD;                 return T_SHAPE; }
".m64n64k64"            { yylval->intVal = ptxSHAPE_646464_MOD;                 return T_SHAPE; }
".m64n72k64"            { yylval->intVal = ptxSHAPE_647264_MOD;                 return T_SHAPE; }
".m64n80k64"            { yylval->intVal = ptxSHAPE_648064_MOD;                 return T_SHAPE; }
".m64n88k64"            { yylval->intVal = ptxSHAPE_648864_MOD;                 return T_SHAPE; }
".m64n96k64"            { yylval->intVal = ptxSHAPE_649664_MOD;                 return T_SHAPE; }
".m64n104k64"           { yylval->intVal = ptxSHAPE_6410464_MOD;                return T_SHAPE; }
".m64n112k64"           { yylval->intVal = ptxSHAPE_6411264_MOD;                return T_SHAPE; }
".m64n120k64"           { yylval->intVal = ptxSHAPE_6412064_MOD;                return T_SHAPE; }
".m64n128k64"           { yylval->intVal = ptxSHAPE_6412864_MOD;                return T_SHAPE; }
".m64n136k64"           { yylval->intVal = ptxSHAPE_6413664_MOD;                return T_SHAPE; }
".m64n144k64"           { yylval->intVal = ptxSHAPE_6414464_MOD;                return T_SHAPE; }
".m64n152k64"           { yylval->intVal = ptxSHAPE_6415264_MOD;                return T_SHAPE; }
".m64n160k64"           { yylval->intVal = ptxSHAPE_6416064_MOD;                return T_SHAPE; }
".m64n168k64"           { yylval->intVal = ptxSHAPE_6416864_MOD;                return T_SHAPE; }
".m64n176k64"           { yylval->intVal = ptxSHAPE_6417664_MOD;                return T_SHAPE; }
".m64n184k64"           { yylval->intVal = ptxSHAPE_6418464_MOD;                return T_SHAPE; }
".m64n192k64"           { yylval->intVal = ptxSHAPE_6419264_MOD;                return T_SHAPE; }
".m64n200k64"           { yylval->intVal = ptxSHAPE_6420064_MOD;                return T_SHAPE; }
".m64n208k64"           { yylval->intVal = ptxSHAPE_6420864_MOD;                return T_SHAPE; }
".m64n216k64"           { yylval->intVal = ptxSHAPE_6421664_MOD;                return T_SHAPE; }
".m64n224k64"           { yylval->intVal = ptxSHAPE_6422464_MOD;                return T_SHAPE; }
".m64n232k64"           { yylval->intVal = ptxSHAPE_6423264_MOD;                return T_SHAPE; }
".m64n240k64"           { yylval->intVal = ptxSHAPE_6424064_MOD;                return T_SHAPE; }
".m64n248k64"           { yylval->intVal = ptxSHAPE_6424864_MOD;                return T_SHAPE; }
".m64n256k64"           { yylval->intVal = ptxSHAPE_6425664_MOD;                return T_SHAPE; }
".m64n8k256"            { yylval->intVal = ptxSHAPE_6408256_MOD;                return T_SHAPE; }
".m64n16k256"           { yylval->intVal = ptxSHAPE_6416256_MOD;                return T_SHAPE; }
".m64n24k256"           { yylval->intVal = ptxSHAPE_6424256_MOD;                return T_SHAPE; }
".m64n32k256"           { yylval->intVal = ptxSHAPE_6432256_MOD;                return T_SHAPE; }
".m64n48k256"           { yylval->intVal = ptxSHAPE_6448256_MOD;                return T_SHAPE; }
".m64n64k256"           { yylval->intVal = ptxSHAPE_6464256_MOD;                return T_SHAPE; }
".m64n80k256"           { yylval->intVal = ptxSHAPE_6480256_MOD;                return T_SHAPE; }
".m64n96k256"           { yylval->intVal = ptxSHAPE_6496256_MOD;                return T_SHAPE; }
".m64n112k256"          { yylval->intVal = ptxSHAPE_64112256_MOD;               return T_SHAPE; }
".m64n128k256"          { yylval->intVal = ptxSHAPE_64128256_MOD;               return T_SHAPE; }
".m64n144k256"          { yylval->intVal = ptxSHAPE_64144256_MOD;               return T_SHAPE; }
".m64n160k256"          { yylval->intVal = ptxSHAPE_64160256_MOD;               return T_SHAPE; }
".m64n176k256"          { yylval->intVal = ptxSHAPE_64176256_MOD;               return T_SHAPE; }
".m64n192k256"          { yylval->intVal = ptxSHAPE_64192256_MOD;               return T_SHAPE; }
".m64n208k256"          { yylval->intVal = ptxSHAPE_64208256_MOD;               return T_SHAPE; }
".m64n224k256"          { yylval->intVal = ptxSHAPE_64224256_MOD;               return T_SHAPE; }
".m64n240k256"          { yylval->intVal = ptxSHAPE_64240256_MOD;               return T_SHAPE; }
".m64n256k256"          { yylval->intVal = ptxSHAPE_64256256_MOD;               return T_SHAPE; }

".row"                  { yylval->intVal = ptxLAYOUT_ROW_MOD;                   return T_LAYOUT; }
".col"                  { yylval->intVal = ptxLAYOUT_COL_MOD;                   return T_LAYOUT; }
".transA"               { yylval->intVal = ptxTRANSA_MOD;                       return T_TRANSAMOD; }
".negA"                 { yylval->intVal = ptxNEGA_MOD;                         return T_NEGAMOD; }
".transB"               { yylval->intVal = ptxTRANSB_MOD;                       return T_TRANSBMOD; }
".negB"                 { yylval->intVal = ptxNEGB_MOD;                         return T_NEGBMOD; }
".ignoreC"              { yylval->intVal = ptxIGNOREC_MOD;                      return T_IGNORECMOD; }


 /* these tokens are identified in setModifier to avoid string leak */
 /* ".regoffset"        { yylval->intVal = ptxSPFORMAT_REGOFFSET_MOD;           return T_SPFORMAT; }
  * ".sp"               { yylval->intVal = ptxSPARSITY_MOD;                     return T_SPARSITY; }
  * ".tf32"             { yylval->intVal = ptxTYPE_TF32_MOD;                    return T_TYPEMOD; }
  * ".bf16"             { yylval->intVal = ptxTYPE_BF16_MOD;                    return T_TYPEMOD; }
  * ".mbarrier"         { yylval->intVal = ptxMBARRIER_MOD;                     return T_MBARRIER; }
  * ".mmio"             { yylval->intVal = ptxMMIO_MOD;                         return T_ORDERMOD; }
  * ".multicast"        { yylval->intVal = ptxMULTICAST_MOD;                    return T_MULTICASTMOD; }
  * ".im2col"           { yylval->intVal = ptxIM2COL_MOD;                       return T_IM2COLMOD; }
  * ".packed_offsets"    { yylval->intVal = ptxPACKEDOFF_MOD;                    return T_PACKEDOFF; }
  */

".64B"                  { yylval->intVal = ptxPREFETCHSIZE_64B_MOD;             return T_PREFETCHSIZE; }
".128B"                 { yylval->intVal = ptxPREFETCHSIZE_128B_MOD;            return T_PREFETCHSIZE; }
".256B"                 { yylval->intVal = ptxPREFETCHSIZE_256B_MOD;            return T_PREFETCHSIZE; }

".L2::64B"              { yylval->intVal = ptxCACHEPREFETCH_L2_64B_MOD;         return T_CACHEPREFETCH; }
".L2::128B"             { yylval->intVal = ptxCACHEPREFETCH_L2_128B_MOD;        return T_CACHEPREFETCH; }
".L2::256B"             { yylval->intVal = ptxCACHEPREFETCH_L2_256B_MOD;        return T_CACHEPREFETCH; }

".texref"               {
                            ptxSymbol t = ptxLookupOpaque(gblState->globalSymbolTable, ".texref", True);

                            yylval->typeVal = t->type;
                            return T_TYPENAME;
                        }
".samplerref"           {
                            ptxSymbol t = ptxLookupOpaque(gblState->globalSymbolTable, ".samplerref", True);

                            yylval->typeVal = t->type;
                            return T_TYPENAME;
                        }
".surfref"              {
                            ptxSymbol t = ptxLookupOpaque(gblState->globalSymbolTable, ".surfref", True);

                            yylval->typeVal = t->type;
                            return T_TYPENAME;
                        }
".width"                        { yylval->intVal = ptxQUERY_WIDTH_MOD;          return T_QUERYMOD;  }
".height"                       { yylval->intVal = ptxQUERY_HEIGHT_MOD;         return T_QUERYMOD;  }
".depth"                        { yylval->intVal = ptxQUERY_DEPTH_MOD;          return T_QUERYMOD;  }
".channel_data_type"            { yylval->intVal = ptxQUERY_CHTYPE_MOD;         return T_QUERYMOD;  }
".channel_order"                { yylval->intVal = ptxQUERY_CHORDER_MOD;        return T_QUERYMOD;  }
".normalized_coords"            { yylval->intVal = ptxQUERY_NORM_MOD;           return T_QUERYMOD;  }
".filter_mode"                  { yylval->intVal = ptxQUERY_FILTER_MOD;         return T_QUERYMOD;  }
".addr_mode_0"                  { yylval->intVal = ptxQUERY_ADDR0_MOD;          return T_QUERYMOD;  }
".addr_mode_1"                  { yylval->intVal = ptxQUERY_ADDR1_MOD;          return T_QUERYMOD;  }
".addr_mode_2"                  { yylval->intVal = ptxQUERY_ADDR2_MOD;          return T_QUERYMOD;  }
".force_unnormalized_coords"    { yylval->intVal = ptxQUERY_UNNORM_MOD;         return T_QUERYMOD;  }
".array_size"                   { yylval->intVal = ptxQUERY_ARRSIZE_MOD;        return T_QUERYMOD;  }
".num_mipmap_levels"            { yylval->intVal = ptxQUERY_MIPLEVEL_MOD;       return T_QUERYMOD;  }
".num_samples"                  { yylval->intVal = ptxQUERY_SAMPLES_MOD;        return T_QUERYMOD;  }
".memory_layout"                { yylval->intVal = ptxQUERY_LAYOUT_MOD;         return T_QUERYMOD;  }

 /* deprecated state space, replaced by .texref */
".tex"                  { return T_TEX;         }


"_"                     { return T_SINK;        }

"call"                  { 
                            yylval->stringVal = stdCOPYSTRING(yytext);
                            return T_CALL;
                        }

{xxid}                  { 
                            yylval->stringVal = stdCOPYSTRING(yytext);
                            return T_IDENT;
                        }

 /* all tokens beginning with '.' should precede {dotid}. */
{dotid}                 { 
                            yylval->stringVal = stdCOPYSTRING(yytext);
                            return T_DOTIDENT;
                        }

{digit}+\.{digit}*{exponent}? {
                                errno = 0;
                                yylval->doubleVal = strtod(yytext, NULL);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return T_FLOAT64;
                        }
\.{digit}+{exponent}?   {
                                errno = 0;
                                yylval->doubleVal = strtod(yytext, NULL);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return T_FLOAT64;
                        }

{digit}+{exponent}      {
                                errno = 0;
                                yylval->doubleVal = strtod(yytext, NULL);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return T_FLOAT64;
                        }

0{octdigit}*U?          {
                                Bool isSigned;

                                errno      = 0;
                                atox(yytext, 8, &yylval->int64Val, &isSigned);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return (isSigned ? T_INT64 : T_UINT64);
                        }

0[xX]{hexdigit}+U?      {
                                Bool isSigned;

                                errno      = 0;
                                atox(yytext+2, 16, &yylval->int64Val, &isSigned);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return (isSigned ? T_INT64 : T_UINT64);
                        }

0[bB]{bit}+U?           {
                                Bool isSigned;

                                errno      = 0;
                                atox(yytext+2, 2, &yylval->int64Val, &isSigned);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return (isSigned ? T_INT64 : T_UINT64);
                        }

{nonzerodigit}{digit}*U? {
                                Bool isSigned;

                                errno      = 0;
                                atox(yytext, 10, &yylval->int64Val, &isSigned);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                return (isSigned ? T_INT64 : T_UINT64);
                        }

0[dD]({hexdigit}{16})   {
                                Bool isSigned;
                                uInt64 x;
                                atox(yytext+2, 16, &x, &isSigned);
                                {
                                   union {
                                     uInt64 i;
                                     Double f;
                                   } v;
                                   v.i = x;
                                   yylval->doubleVal = v.f;
                                }
                                return T_FLOAT64;
                        }

0[fF]({hexdigit}{8})    {
                                uInt32 x= strtoul(yytext+2, NULL, 16);
                                stdCHECK_WITH_POS(errno != ERANGE, (ptxMsgConstantOverflow, ptxsLwrPos(gblState), yytext));
                                {
                                   union {
                                     uInt32 i;
                                     Float f;
                                   } v;
                                   v.i = x;
                                   yylval->floatVal = v.f;
                                }
                                return T_FLOAT;
                        }

{string}                {       return unrecognizedString(&yylval->stringVal, yytext, yyleng); }

{sourcePos}({ws}+{digit})*{ws}*/{newline}            { 
                                Char *start, *end;
                                Int  lineno;

                                start= yytext+1;
                                while (isspace(*start)) { start++; }
                                if (*start == 'l') { start += 4; }

                                lineno= strtoul(start,&start,10);
                                while (*(start++) != '"') {}

                                end= start-1;
                                while (*(++end) != '"') {}
                               *end= 0;

                                ptxset_lineno(lineno - 1, gblState->scanner);
                                gblState->parseData->ptxfilename = stdCOPYSTRING(start);
                                msgCppUpdateSourceStruct( gblState->ptxFileSourceStruct, ptxget_lineno(gblState->scanner), start, lineno, end+1 );
                            }

"@@DWARF"               { return T_DWARF;    }
"@progbits"             { return T_PROGBITS; }

{ws}*                   { }
"/*"                    { skip_until(gblState, "*/",False,input,False); }
"//"                    { skip_until(gblState, "\n",False,input,True); }

{newline}               { ptxResetNumScopesOnLine(gblState->parseData);}

.                       { return yytext[0]; }

<<EOF>>                 {
                            if (gblState->parseData->macro_stack_ptr == 0) {
                                yyterminate();
                            } else {
                                if (gblState->parseData->pushedSourcePos) {
                                    restoreSourcePos(gblState);
                                }
                                ptx_delete_buffer( YY_LWRRENT_BUFFER, yyscanner);
                                ptx_switch_to_buffer( gblState->parseData->macro_stack[--gblState->parseData->macro_stack_ptr], yyscanner );
                                if (gblState->parseData->macro_stack_ptr == 0) {
                                    // store final expansion of UserPTX
                                    CT_DEBUG_DO("macro_expansion", 1, storeFinalExpansionOfUserMacro(gblState););
                                }
                            }
                        }

%%

// We need to put these functions at the end of file so that flex generated types
// like yyguts_t are available

void ptxPushMacroBody( String s,  String fileName, uInt lineNo, ptxParsingState parseState, void *scanner)
{
    // TODO: Implement this function using 'ptx_push_buffer' instead of explicitly
    //       managing macro_stack
    struct yyguts_t * yyg = (struct yyguts_t*)scanner;
    if (parseState->parseData->macro_stack_ptr >= MAX_MACRO_DEPTH) {
        ptxerror(scanner, parseState, "macro nesting too deep!");
    }
    parseState->parseData->macro_stack[parseState->parseData->macro_stack_ptr++] = (void *) YY_LWRRENT_BUFFER;

    CT_DEBUG_DO("macro_expansion", 1, generateMacroExpansionDetails(s,
                                            parseState->parseData->macro_stack_ptr,
                                            (( YY_BUFFER_STATE) parseState->parseData->macro_stack[parseState->parseData->macro_stack_ptr - 1])->yy_ch_buf, parseState););

    int lwrLineNo = ptxget_lineno(parseState->scanner);
    ptx_scan_string(s, scanner);
    ptxset_lineno(lwrLineNo, parseState->scanner);

    stdFREE(s);

    ptxPushSourcePos(parseState, fileName, lineNo);
}

void ptxInitLexState(ptxParsingState parseState)
{
    // We need to ensure that buffer is created before setting current sourcePos
    struct yyguts_t * yyg;
    ptxlex_init(&parseState->scanner);
    yyg = (struct yyguts_t*)parseState->scanner;
	if ( ! YY_LWRRENT_BUFFER ) {
	    ptxensure_buffer_stack (parseState->scanner);
		YY_LWRRENT_BUFFER_LVALUE =
				ptx_create_buffer(yyin,YY_BUF_SIZE ,parseState->scanner);
    }
    ptxInitLwrrentSourcePos(parseState, parseState->inputFileName, 1);
    // We need to set extra parameter inorder to access in in YY_INPUT
    ptxset_extra((void *) parseState, parseState->scanner);
}

void ptxDestroyLexState(ptxParsingState parseState)
{
    ptxlex_destroy(parseState->scanner);
}

void ptxerror(void *scanner, ptxParsingState parseState, String message)
{
    if (parseState->ptxObfuscation) {
        msgReport (ptxMsgParsingObfuscatedError);
    } else {
        msgReportWithPos(ptxMsgParsingError, ptxsLwrPos(parseState), ptxget_text(scanner), message );
    }
}

