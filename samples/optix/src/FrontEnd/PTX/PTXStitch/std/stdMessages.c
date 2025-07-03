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
 *  Module name              : stdMessages.c
 *
 *  Description              :
 *
 *         This module allows for printing messages of different
 *         severity levels. Messages are printed by message identifier,
 *         with message texts defined in a central message description file.
 *         In this way, some localization is possible; message texts are
 *         isolated, and can be changed without digging through the tools,
 *         and also by this central message definition it is more
 *         easy to document the different messages that a specific tool
 *         can generate. Also, using this module, some uniformity can be
 *         achieved in the layout and contents of messages printed by
 *         a range of libraries and tools.
 *
 *         Message definitions are expected in a file called <prefix>MessageDefs.h,
 *         which must contain a triple MSG(msg, level, format) for each defined
 *         message. The format is a usual format string as used in printf.
 *         The following is a sample message definition:
 *
 *              MSG( cmdoptMsgNotABool,  Fatal,  "'%s': expected true or false" )
 *         
 *         Message texts are printed using the stdSYSLOG functions defined in stdLocal,
 *         prefixed with a string indicating the level. Because messages are
 *         often used in command line tools, the name of this tool can be specified,
 *         to be included in the message prefix.
 */

/*------------------------------- Includes -----------------------------------*/

#include "stdLocal.h"
#include "stdMessages.h"
#include "stdMessageDefs.h"
#include "stdString.h"
#include "stdStdFun.h"
#include "stdMap.h"

/*------------------------------ Definitions ---------------------------------*/

static cString levelRepresentations[]= { Nil, 
                                        "", 
                                        "info    ", 
                                        "warning ", 
                                        "error*  ", 
                                        "error   ", 
                                        "fatal   " };
                                        
static const uInt8  channelIndex[]= { 4, 
                                0, 
                                0, 
                                1, 
                                1, 
                                2, 
                                4 };

static stdWriter_t channelWriters[5];

// Move below variables into thread context if any client needs to set different
// value
static const Bool   msgColonBeforeLevel = False;
static const Bool   msgDontEchoNonPosMessages = False;


#if defined(GPU_DRIVER_SASSLIB)
struct msgMessageStruct m_lastErrorMessage = {msgInfo,True,False,NULL};
msgMessage lastErrorMessage = &m_lastErrorMessage;
#endif

/*------------------------------- API Functions ------------------------------*/

/*
 * Function        : Create new handle to record the structure of parsed input file
 * Parameters      : fileName  (I) Top level source file to parse
 * Function Result : Requested new structure handle, flags default to msgEchoMessageFlag
 */
msgSourceStructure_t STD_CDECL msgCreateSourceStructure( cString fileName )
{
    msgSourceStructure_t result;
    
    stdNEW(result);
    result->flags= msgEchoMessageFlag;
    
    msgPushInclude(result,0,fileName);
    
    result->top = result->lwrNode;

    return result;
}

/*
 * Function        : Set mode flags in source structure
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   flags        (I)  New flags to set
 * Function Result : 
 */
void STD_CDECL msgSetSourceStructureFlags( msgSourceStructure_t sourceStruct, uInt32 flags )
{
    sourceStruct->flags= flags;
}

/*
 * Function        : Delete previously create structure handle
 * Parameters      : sourceStruct (I) Source structure handle to delete
 * Function Result : -
 */
 
   /*
    * Note: leave the inclusion structure in place,
    *       so that the source positions referring to this
    *       structure can still be used for generating messages.
    */
    
    static void STD_CDECL deleteFileMessage( msgFileMessage message )
    {
        stdFREE(message->text);
        stdFREE(message);
    }

    static void STD_CDECL deleteFileNode( msgFileNode node )
    {
        stdFREE(node->fileName);

        listTraverse( node->includes, (stdEltFun)deleteFileNode,      Nil );
        listTraverse( node->messages, (stdEltFun)deleteFileMessage,   Nil );

        listDelete( node->messages ); node->messages= Nil;
    }

void STD_CDECL msgDeleteSourceStructure( msgSourceStructure_t sourceStruct )
{
    deleteFileNode( sourceStruct->top );

    sourceStruct->flags &= ~msgRecordMessageFlag;
    sourceStruct->flags |=  msgEchoMessageFlag;

    if (sourceStruct->lwrFileNode) {
        sourceStruct->lwrFileNode= Nil;
        if (sourceStruct->lwrFileIndex) { 
            mapDelete(sourceStruct->lwrFileIndex);
            fclose   (sourceStruct->lwrFile);
        }
    }
}

/*
 * Function        : Change source position on top of include stack to new source position
 * Parameters      : sourceStruct (I)  Source structure handle to set
 *                   sourcePos    (I)  Source position to set handle to
 * Function Result : requested source position
 */
void STD_CDECL msgSetSourcePos( msgSourceStructure_t sourceStruct, msgSourcePos_t sourcePos )
{
    sourceStruct->lwrPos = sourcePos;
}

/*
 * Function        : Obtain source position marker from current top of include stack,
 *                   indicating specified line number
 * Parameters      : sourceStruct (I) Source structure handle to pull error from
 *                   lineNo       (I) Line number in requested source position
 * Function Result : requested source position
 */
msgSourcePos_t STD_CDECL msgPullSourcePos( msgSourceStructure_t sourceStruct, uInt lineNo )
{
    if ( sourceStruct->lwrPos && sourceStruct->lwrPos->lineNo == lineNo ) {
        return sourceStruct->lwrPos;
    } else {
    
       stdNEW(sourceStruct->lwrPos);
       sourceStruct->lwrPos->lineNo = lineNo;
       sourceStruct->lwrPos->file   = sourceStruct->lwrNode;

       return sourceStruct->lwrPos;
   }
}

/*
 * Function        : Push a new file include marker, and make it current
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   lineNo       (I)  last active line number of the 'old' file before the new file becomes active
 *                   fileName     (I)  Name of included file
 * Function Result : 
 */
void STD_CDECL msgPushInclude( msgSourceStructure_t sourceStruct, uInt lineNo, cString fileName )
{
    msgFileNode parent= sourceStruct->lwrNode;
        
    stdNEW(sourceStruct->lwrNode);
    
    sourceStruct->lwrNode->fileName      = stdCOPYSTRING(fileName);
    sourceStruct->lwrNode->parent.file   = parent;
    sourceStruct->lwrNode->parent.lineNo = lineNo;
    sourceStruct->lwrNode->sourceStruct  = sourceStruct;
    
    sourceStruct->lwrPos = Nil;
    
    if (parent) {
        sourceStruct->lwrNode->level = parent->level+1;
        listAddTo(sourceStruct->lwrNode, (Pointer)&parent->includes );
    }
}

/*
 * Function        : Pop top of file include marker stack
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 * Function Result : 
 */
void STD_CDECL msgPopInclude( msgSourceStructure_t sourceStruct )
{
    if (sourceStruct->lwrNode->parent.file) {
        sourceStruct->lwrNode= sourceStruct->lwrNode->parent.file;
        sourceStruct->lwrPos = Nil;
    }
}


/*------------------------------- API Functions ------------------------------*/

/*
 * Function        : Update source structure with cpp directives.
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   oldLineNo    (I)  Lwrrently active line number
 *                   fileName     (I)  File name in directive
 *                   lineNo       (I)  Line number in directive
 *                   cppDirectives(I)  Trailer of cpp directives
 * Function Result : 
 */
void STD_CDECL msgCppUpdateSourceStruct( msgSourceStructure_t sourceStruct, uInt oldLineNo, cString fileName, uInt lineNo, cString cppDirectives )
{
    Char *endptr;
    uInt next= strtoll(cppDirectives,&endptr,0);
    
    if (endptr == cppDirectives || next == 3) {
        stdFREE(sourceStruct->lwrNode->fileName);
        sourceStruct->lwrNode->fileName = stdCOPYSTRING(fileName);
    } else 
    if (next == 1) {
        stdASSERT( lineNo == 1, ("Unexpected cpp line directive") );
        msgPushInclude( sourceStruct, oldLineNo, fileName );
    
    } else
    if (next == 2) {
        msgPopInclude(sourceStruct);
        stdASSERT( stdEQSTRING(sourceStruct->lwrNode->fileName,fileName), ("Unexpected cpp line directive") );
    }
}

/*------------------------------- API Functions ------------------------------*/

/*
 * Function        : Decide on the relative parsing order of the specified two source
 *                   positions in their parsing tree.
 * Parameters      : l,r          (I)  Source positions to compare
 * Function Result : l less than, or equal to r.
 */
Bool STD_CDECL msgSourcePosLessEq( msgSourcePos_t l, msgSourcePos_t r)
{
   while (l->file->level > r->file->level) { l= &l->file->parent; }
   while (l->file->level < r->file->level) { r= &r->file->parent; }
   
   while (l->file != r->file) { 
       l= &l->file->parent; 
       r= &r->file->parent;
   }
   
   return l->lineNo <= r->lineNo;
}

/*------------------------------- API Functions ------------------------------*/

/*
 * Function        : Generate a list file with all reported errors.
 * Parameters      : sourceStruct (I)  Source structure handle to list
 *                   fileName     (I)  Name of list file to write
 * Function Result : 
 */
void STD_CDECL msgGenSourceListing( msgSourceStructure_t sourceStruct, cString fileName )
{
    stdASSERT( False, ("msgGenSourceListing is not implemented yet") );
}

/*------------------------------- API Functions ------------------------------*/

    #define INDEX_GRANULARITY 10

    static void getFileIndex( msgSourceStructure_t sourceStruct, cString fileName )
    {
        FILE *file= fopen(fileName,"r");
        
        if (!file) {
            sourceStruct->lwrFileIndex= Nil;
        } else {
            Int c = getc(file);
            Int lineCount= 0;
            
            sourceStruct->lwrFile      = file;
            sourceStruct->lwrFileIndex = mapNEW(uInt,1024);
            
            while (c!=EOF) {
                while (c!='\n' && c!=EOF) { c = getc(file); }
                if (c=='\n') {
                    lineCount++;
                    
                    if ((lineCount % INDEX_GRANULARITY)==0) {
                        mapDefine( 
                            sourceStruct->lwrFileIndex, 
                            (Pointer)(Address)(lineCount / INDEX_GRANULARITY), 
                            (Pointer)(Address)ftell(file)
                        );
                    }
                    c = getc(file);
                }
            }   
        }
    }

    static String readLine( FILE *file )
    {
        if (feof(file)) {
            return Nil;
        } else {
            stdString_t buffer= stringNEW();
            Int         c     = getc(file);
            
            stringAddBuf(buffer,"# ");
            
            while (c != EOF && c != '\n') {
                stringAddChar(buffer,c);
                c = getc(file);
            }
            
            stringAddChar(buffer,'\n');
            
            return stringStripToBuf(buffer);
        }
    }

    static String getFileLine( msgSourcePos_t pos )
    {
        msgFileNode           file         = pos->file;
        msgSourceStructure_t  sourceStruct = file->sourceStruct;
        
        if (sourceStruct->lwrFileNode != file) {
            if (sourceStruct->lwrFileIndex) { 
                mapDelete(sourceStruct->lwrFileIndex);
                fclose   (sourceStruct->lwrFile);
            }
            sourceStruct->lwrFileNode = file;
            getFileIndex(sourceStruct,file->fileName);
        }
        
        if (sourceStruct->lwrFileIndex) { 
            uInt lineNo      = pos->lineNo - 1; // zero-origined
            uInt indexPos    = (Address)mapApply(sourceStruct->lwrFileIndex,(Pointer)(Address) (lineNo / INDEX_GRANULARITY) );
            uInt indexOffset = lineNo % INDEX_GRANULARITY;
            
            if (fseek(sourceStruct->lwrFile,indexPos,SEEK_SET)==0) {
                String result;
            
                do {
                    result = readLine(sourceStruct->lwrFile);
                } while (indexOffset-- != 0);
                
                if (result) { return result; }
            }
        }
        
        return stdCOPYSTRING(""); 
    }

    static cString getMessageErrorClassPrefix( msgMessageLevel level )
    {
        if (msgGetAddErrorClassPrefixes()) {
            switch (level) {
            case msgNInfo   : return msgInfoClassPrefix;   
            case msgInfo    : return msgOutputClassPrefix; 
            case msgWarning : return msgWarningClassPrefix;
            case msgWError  : return msgErrorClassPrefix;  
            case msgError   : return msgErrorClassPrefix;  
            case msgFatal   : return msgErrorClassPrefix;  
            default         : stdASSERT( False, ("Case label out of bounds") );
            }
        } 
        return "";
    }

    static String getMessageLine( msgMessage message, msgMessageLevel level, msgSourcePos_t pos, cString msgText )
    {
        stdString_t buffer= stringNEW();
        String      prefix;
        uInt        tabstop;

        stringAddBuf(buffer, getMessageErrorClassPrefix(level) ); 

        if (msgGetToolName()) {
            stringAddFormat(buffer,"%s", msgGetToolName()); 
            stringAddFormat(buffer," ");
        }

        prefix = stringToBuf(buffer);

        if (pos && pos->file && pos->lineNo != msgINF_LINE_NUMBER) {
            stringAddFormat(buffer,"%s, line %d; ", pos->file->fileName, pos->lineNo);
        }

        stringAddFormat(buffer,"%s%s", (msgColonBeforeLevel?": ":""), levelRepresentations[level]);
        tabstop = stringSize(buffer) - strlen(prefix);
        
        stringAddFormat(buffer,": ");
        
        while (*msgText) {
            Char c= *(msgText++);
        
            stringAddChar(buffer,c);
        
            if ( c == '\n'
              && !msgGetDontPrefixErrorLines()
               ) {
                uInt i;
                stringAddBuf(buffer,prefix);
                for (i=0; i<tabstop; i++) { stringAddChar(buffer,' '); }
                stringAddBuf(buffer,". ");
            }
        }
        
        if (msgGetSuffix()) {
            stringAddFormat(buffer," %s",msgGetSuffix()); 
        }
        stringAddChar(buffer,'\n');

        stdFREE(prefix);
        
        return stringStripToBuf(buffer);
    }


/*
 * Function        : Report Message with an optional file source position and
 *                   arguments, and exit with status -1
 *                   in case the message's level is Fatal.
 * Parameters      : message (I) name of a message that has been
 *                               specified in local file <prefix>MessageDefs.h
 *                   pos     (I) a file source position to report the message at, or Nil
 *                   ...     (I) arguments to be filled in into the representation
 *                               string corresponding with 'message' in <prefix>MessageDefs.h.
 * Function Result : -
 */
#define MSGMAXLEN (100*1024)
static void msgVReportWithPos( msgMessage message, msgSourcePos_t pos, va_list ap )
{
    if (!message->disabled) {
        Bool              echoMessage     = (!pos && !msgDontEchoNonPosMessages) || (pos && !pos->file->sourceStruct) || (pos && pos->file && pos->file->sourceStruct && pos->file->sourceStruct->flags & msgEchoMessageFlag);
        Bool              recordMessage   =                                                                     (pos && pos->file && pos->file->sourceStruct && pos->file->sourceStruct->flags & msgRecordMessageFlag);
        Bool              echoMessageLine =                                                                     (pos && pos->file && pos->file->sourceStruct && pos->file->sourceStruct->flags & msgEchoMessageLineFlag);

        msgMessageLevel   level           = message->level;

        stdString_t       buffer;
        String            msgText;
        String            msgLine = NULL;

        if ( level == msgWarning )
        {
            if (msgGetIgnoreWarnings()) { level= msgNoReport; } else
            if (msgGetWarnAsError()   ) { level= msgError;    }
        }

        if (message == stdMsgMemoryOverFlow) {
            // Avoid infinite failed-malloc message loops
            // by directly calling fatal error print functions:

            if (msgGetToolName()) {
                stdFSYSLOG(msgGetToolName()); 
                stdFSYSLOG(" ");
            }

            stdFSYSLOG("%s%s", (msgColonBeforeLevel?": ":""), levelRepresentations[level]);
            stdFSYSLOG(": ");
            stdVFSYSLOG(message->repr,ap);
            stdFSYSLOG("\n");
            stdGetThreadContext()->errorsFound = True;
            msgRaise(message);
            return;
            
        } else 
        if (level != msgNoReport) {

            buffer  = stringNEW();
                      stringAddVFormat(buffer,message->repr,ap);
            msgText = stringStripToBuf(buffer);
            msgLine = getMessageLine(message, level, pos, msgText);

            if (echoMessage) {
                String      fileLine   = echoMessageLine ? getFileLine(pos) : S("");
                uInt        channelIdx = channelIndex  [level];
                stdWriter_t channel    = channelWriters[channelIdx];

                if (channel) {
                    wtrPrintf(channel, "%s%s", fileLine, msgLine);
                } else {
                    stdSYSLOG(         "%s%s", fileLine, msgLine);
                }
                
                if (echoMessageLine) { stdFREE(fileLine); }
                //stdFREE(msgLine);
            }

            if (recordMessage) {
                msgFileMessage recorded;

                stdNEW(recorded);

                recorded->lineNo  = pos->lineNo;
                recorded->message = message;
                recorded->text    = msgText;

                listAddTo(recorded, (Pointer)&pos->file->messages);
            } else {
#ifdef COPIED_MESSAGES
                if (message->copied) {
                    // Free doesn't take cString so cast back to String
                    stdFREE2((String)message->repr,message);
                }
#else
                // gpgpucomp never copies messages,
                // and doing so is dangerous and confuses coverity
                // since would be error if message was reused,
                // so just assert that this should not happen.
                // We leave possible support under if-def
                // because other codes like assembler and graphics
                // also use this.
                stdASSERT( message->copied == False, ("message->copied not supported") );
#endif
                
                stdFREE(msgText);
            }
        }
        
        if (level >= msgWarning) {
            stdGetThreadContext()->warningsFound = True;
        }
        if (level >= msgError) {
            stdGetThreadContext()->errorsFound = True;
#if defined(GPU_DRIVER_SASSLIB)
            //// SASS: accumulate current message into lastErrorMessage
            lastErrorMessage->level = message->level;
            if (lastErrorMessage->repr) {
                cString oldString = lastErrorMessage->repr;
                size_t origLen = strlen(oldString);
                size_t msgLen = strlen(msgLine);
                if ((origLen + msgLen + 2) <= MSGMAXLEN) {
                    // Do not use stdMalloc here, since Jas will sometimes attempt to clean up all its memory and end up deleting the message itself as well
                    String tmp = malloc(origLen + msgLen + 3);
                    strcpy(tmp, oldString);
                    strcpy(tmp + origLen, "\n\n");
                    strcpy(tmp + origLen + 2, msgLine);
                    lastErrorMessage->repr = (cString)tmp;
                    stdFREE((String)oldString);
                }
            } else {
                lastErrorMessage->repr = stdCOPYSTRING(msgLine);
            }
#endif
        }

        if (msgLine) {
            stdFREE(msgLine);
        }
#if defined(GPU_DRIVER_SASSLIB)
        if (level == msgFatal) {
            msgRaise(lastErrorMessage);
#else
        if (level >= msgFatal) {
            msgRaise(message);
#endif
        }
    }
}


/*
 * Function        : Report Message with arguments, and exit with status -1
 *                   in case the message's level is Fatal.
 * Parameters      : message (I) name of a message that has been
 *                               specified in local file <prefix>MessageDefs.h
 *                   pos     (I) a file source position to report the message at, or Nil
 *                   ...     (I) arguments to be filled in into the representation
 *                               string corresponding with 'message' in <prefix>MessageDefs.h.
 * Function Result : -
 */
void STD_CDECL msgReport( msgMessage message, ... )
{
    va_list ap;
    va_start (ap, message);
    msgVReportWithPos(message,Nil,ap);
    va_end (ap);  
}


/*
 * Function        : Report Message with an optional file source position and
 *                   arguments, and exit with status -1
 *                   in case the message's level is Fatal.
 * Parameters      : message (I) name of a message that has been
 *                               specified in local file <prefix>MessageDefs.h
 *                   pos     (I) a file source position to report the message at, or Nil
 *                   ...     (I) arguments to be filled in into the representation
 *                               string corresponding with 'message' in <prefix>MessageDefs.h.
 * Function Result : -
 */
void STD_CDECL msgReportWithPos( msgMessage message, msgSourcePos_t pos, ... )
{
    va_list ap;
    va_start (ap, pos);
    msgVReportWithPos(message,pos,ap);
    va_end (ap);  
}



/*
 * Function        : Swap message channels for specific message levels.
 *                   Note: by default, all installed message channels will print via stdSYSLOG.
 *                   Note: the fatal error channel cannot be swapped; fatal errors will always print via stdSYSLOG
 * Parameters      : info,warning,error  (IO)  Pointers to locations of specific error writers, or Nil.
 *                                             When non-Nil, the writer object at the referenced location
 *                                             will be installed as writer for the respective error level,
 *                                             and the previously active writer object will be returned in
 *                                             this location.
 * Note            : The writers are modeled below as Pointers iso stdWriters due to problems related to #include cycles
 */
void STD_CDECL msgSwapMessageChannels( Pointer /*stdWriter_t*/ *info, Pointer /*stdWriter_t*/ *warning, Pointer /*stdWriter_t*/ *error )
{
    if (info   ) { stdSWAP( *info,    channelWriters[0], Pointer ); }
    if (warning) { stdSWAP( *warning, channelWriters[1], Pointer ); }
    if (error  ) { stdSWAP( *error,   channelWriters[2], Pointer ); }
}




uInt STD_CDECL stdSourcePosHash( msgSourcePos_t e )
{
    return stdStringHash(e->file->fileName)
        ^ _stdIntHash   (e->lineNo);
}

Bool STD_CDECL stdSourcePosEqual( msgSourcePos_t e1, msgSourcePos_t e2 )
{
    return stdEQSTRING(e1->file->fileName,e2->file->fileName)
        && (e1->lineNo == e2->lineNo);
}

/*
 * msgGetWarnAsError()           - Obtain the setting which indicates whether warnings should be reported as errors
 *
 */
Bool STD_CDECL msgGetWarnAsError()
{
    return stdGetThreadContext()->mInfo.msgWarnAsError;
}

/*
 * msgSetWarnAsError()           - Changes the behaviour of how warnings are reported.
 *                                 If lMsgWarnAsError is -
 *                                 * True  - Warnings are treated and reported as errors
 *                                 * False - Warnings are treated as warnings
 */
void STD_CDECL msgSetWarnAsError(Bool lMsgWarnAsError)
{
    stdGetThreadContext()->mInfo.msgWarnAsError = lMsgWarnAsError;
}

/*
 * msgGetIgnoreWarnings()        - Obtain the setting which indicates whether warnings are ignored while reporting
 *
 */
Bool STD_CDECL msgGetIgnoreWarnings()
{
    return stdGetThreadContext()->mInfo.msgIgnoreWarnings;
}

/*
 * msgSetIgnoreWarnings()        - Changes the behaviour of reporting of warnings. If lMsgIgnoreWarning is -
 *                                  * True  - Warnings are not reported and are ignored
 *                                  * False - Warnings are reported as warnings
 */
void STD_CDECL msgSetIgnoreWarnings(Bool lMsgIgnoreWarnings)
{
    stdGetThreadContext()->mInfo.msgIgnoreWarnings = lMsgIgnoreWarnings;
}

/*
 * msgGetToolName()              - Obtain the name of the tool using stdlib which is lwrrently being run
 * 
 */
cString STD_CDECL msgGetToolName()
{
    return stdGetThreadContext()->mInfo.msgToolName;
}

/*
 * msgSetToolName()              - Set the name of the tool using stdlib which is lwrrently being run
 *                                 lMsgToolNane :          Name of the tool which is being lwrrently called upon
 *                                                         This string should be caller allocated string and 
 *                                                         should also be freed by the caller code
 * 
 */
void STD_CDECL msgSetToolName(cString lMsgToolName)
{
    /*
     * String to be used as prefix for reported messages.
     * Generally, this is the name of a tool in which this 
     * library is used. The default prefix is the empty string.
     */
    stdGetThreadContext()->mInfo.msgToolName = lMsgToolName;
}

/*
 * msgGetSuffix()                - Obtain the suffix message that is going to be printed along with any error message
 * 
 */
String STD_CDECL msgGetSuffix()
{
    return stdGetThreadContext()->mInfo.msgSuffix;
}

/*
 * msgSetSuffix()                - Set the message that is going to be printed suffixed with any error message
 *                                 This string should be caller allocated string and 
 *                                 should also be freed by the caller code
 * 
 */
void STD_CDECL msgSetSuffix(String lMsgSuffix)
{
    /*
     * String to be used as suffix for reported messages.
     * Generally, this is the target arch.
     * If NULL, then no suffix is added.
     */
    stdGetThreadContext()->mInfo.msgSuffix = lMsgSuffix;
}

/*
 * msgGetAddErrorClassPrefixes() - Obtain the setting which indicates whether to add any 
 *                                 error class prefix before reporting out error
 * 
 */
Bool STD_CDECL msgGetAddErrorClassPrefixes()
{
    return stdGetThreadContext()->mInfo.msgAddErrorClassPrefixes;
}

/*
 * msgSetAddErrorClassPrefixes() - Changes the setting whether to add the error class prefix 
 *                                 before printing out the error or not.
 *                                 If lMsgAddErrorClassPrefixes is -
 *                                   * True  - Error class prefix is added to the error message
 *                                   * False - The error message is printed as is
 *
 */
void STD_CDECL msgSetAddErrorClassPrefixes(Bool lMsgAddErrorClassPrefixes)
{
    stdGetThreadContext()->mInfo.msgAddErrorClassPrefixes = lMsgAddErrorClassPrefixes;
}

/*
 * msgGetDontPrefixErrorLines()  - Returns whether every line in a multi-line error is prefixed 
 *                                 with error class prefix and tool name
 * 
 */
Bool STD_CDECL msgGetDontPrefixErrorLines()
{
    return stdGetThreadContext()->mInfo.msgDontPrefixErrorLines;
}

/*
 * msgSetDontPrefixErrorLines()  - Change the setting of whether everyline in a multi-line error
 *                                 is prefixed with error class prefix and tool name.
 *                                 If lMsgDontPrefixErrorLines is -
 *                                  * True  - Only the first line gets prefixed
 *                                  * False - Every line gets prefixed
 */
void STD_CDECL msgSetDontPrefixErrorLines(Bool lMsgDontPrefixErrorLines)
{
    stdGetThreadContext()->mInfo.msgDontPrefixErrorLines = lMsgDontPrefixErrorLines;
}

/*
 * msgRestoreDontPrefixErrorLines : Restore the old value of msgDontPrefixErrorLines
 *
 */

void STD_CDECL msgRestoreDontPrefixErrorLines()
{
    msgSetDontPrefixErrorLines(stdGetThreadContext()->mInfo.oldMsgDontPrefixErrorLines);
}

/*
 * msgPreserveDontPrefixErrorLines : Preserve the current value of msgDontPrefixErrorLines
 *
 */

void STD_CDECL msgPreserveDontPrefixErrorLines()
{
    stdGetThreadContext()->mInfo.oldMsgDontPrefixErrorLines =  msgGetDontPrefixErrorLines();
}

/*
 * msgRestoreAddErrorClassPrefixes() - Restore the old value of the addErrorClassPrefix
 *
 */

void STD_CDECL msgRestoreAddErrorClassPrefixes()
{
    msgSetAddErrorClassPrefixes(stdGetThreadContext()->mInfo.oldMsgAddErrorClassPrefixes);
}

/*
 * msgPreserveAddErrorClassPrefixes() - Preserve the current value of the addErrorClassPrefix
 *
 */

void STD_CDECL msgPreserveAddErrorClassPrefixes()
{
    stdGetThreadContext()->mInfo.oldMsgAddErrorClassPrefixes =  msgGetAddErrorClassPrefixes();
}
