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
 *  Module name              : stdMessages.h
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

#include "stdLocal.h"
#ifndef stdMessages_INCLUDED
#define stdMessages_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"
#include "setjmp.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------ Definitions ---------------------------------*/

typedef struct msgMessageStruct          *msgMessage;
typedef struct msgFileInclusionStruct    *msgFileInclusion;
typedef struct msgFileNodeStruct         *msgFileNode;
typedef struct msgSourceStructureStruct  *msgSourceStructure_t;
typedef struct msgSourcePosStruct        *msgSourcePos_t;
typedef struct msgFileMessageStruct      *msgFileMessage;


#define msgINF_LINE_NUMBER  0xfffffff

typedef enum {
    msgNoReport,  // This suppresses message reports
    msgNInfo,     // This is an info, but reports messages without prefix.
    msgInfo,
    msgWarning,
    msgWError,    // This is an error, but allowed to pass for compatibility reasons
    msgError,
    msgFatal
} msgMessageLevel;

    
typedef struct msgMessageStruct {
    msgMessageLevel  level;
    Bool             disabled;
    Bool             copied;
    cString          repr;
} msgMessageRec;

typedef struct msgSourcePosStruct {
    msgFileNode           file;
    uInt                  lineNo;
} msgSourcePosRec;


    struct msgFileMessageStruct {
        uInt                  lineNo;
        msgMessage            message;
        String                text;
    };

typedef struct msgFileNodeStruct {
    msgSourcePosRec       parent;
    uInt                  level;
    String                fileName;
    Pointer               includes;  // List
    Pointer               messages;  // List
    msgSourceStructure_t  sourceStruct;
} msgFileNodeRec;


#define msgRecordMessageFlag   0x1
#define msgEchoMessageFlag     0x2
#define msgEchoMessageLineFlag 0x4

struct msgSourceStructureStruct {
    uInt32                flags;
    msgFileNode           top;
    
    msgSourcePos_t        lwrPos;
    msgFileNode           lwrNode;
    
    msgFileNode           lwrFileNode;
    Pointer               lwrFileIndex;   // uInt -> uInt
    FILE                 *lwrFile;
};

#define msgGetFileName(s) \
         ((s)->file->fileName)

/*-------------------------- SourcePos Construction --------------------------*/

/*
 * Function        : Create new handle to record the structure of parsed input file
 * Parameters      : fileName  (I) Top level source file to parse
 * Function Result : Requested new structure handle, flags default to msgEchoMessageFlag
 */
msgSourceStructure_t STD_CDECL msgCreateSourceStructure( cString fileName );

/*
 * Function        : Set mode flags in source structure
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   flags        (I)  New flags to set
 * Function Result : 
 */
void STD_CDECL msgSetSourceStructureFlags( msgSourceStructure_t sourceStruct, uInt32 flags );

/*
 * Function        : Delete previously create structure handle
 * Parameters      : sourceStruct (I) Source structure handle to delete
 * Function Result : -
 */
void STD_CDECL msgDeleteSourceStructure( msgSourceStructure_t sourceStruct );

/*
 * Function        : Change source position on top of include stack to new source position
 * Parameters      : sourceStruct (I)  Source structure handle to set
 *                   sourcePos    (I)  Source position to set handle to
 * Function Result : requested source position
 */
void STD_CDECL msgSetSourcePos( msgSourceStructure_t sourceStruct, msgSourcePos_t sourcePos );

/*
 * Function        : Obtain source position marker from current top of include stack,
 *                   indicating specified line number
 * Parameters      : sourceStruct (I)  Source structure handle to pull error from
 *                   lineNo       (I)  Line number in requested source position
 * Function Result : requested source position
 */
msgSourcePos_t STD_CDECL msgPullSourcePos( msgSourceStructure_t sourceStruct, uInt lineNo );

/*
 * Function        : Push a new file include marker, and make it current
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   lineNo       (I)  last active line number of the 'old' file before the new file becomes active
 *                   fileName     (I)  Name of included file
 * Function Result : 
 */
void STD_CDECL msgPushInclude( msgSourceStructure_t sourceStruct, uInt lineNo, cString fileName );

/*
 * Function        : Pop top of file include marker stack
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 * Function Result : 
 */
void STD_CDECL msgPopInclude( msgSourceStructure_t sourceStruct );

/*
 * Function        : Generate a list file with all reported errors.
 * Parameters      : sourceStruct (I)  Source structure handle to list
 *                   fileName     (I)  Name of list file to write
 * Function Result : 
 */
void STD_CDECL msgGenSourceListing( msgSourceStructure_t sourceStruct, cString fileName );

/*
 * Function        : Update source structure with cpp directives.
 * Parameters      : sourceStruct (I)  Source structure handle to affect
 *                   oldLineNo    (I)  Lwrrently active line number
 *                   fileName     (I)  File name in directive
 *                   lineNo       (I)  Line number in directive
 *                   cppDirectives(I)  Trailer of cpp directives
 * Function Result : 
 */
void STD_CDECL msgCppUpdateSourceStruct( msgSourceStructure_t sourceStruct, uInt oldLineNo, cString fileName, uInt lineNo, cString cppDirectives );


/*
 * Function        : Decide on the relative parsing order of the specified two source
 *                   positions in their parsing tree.
 * Parameters      : l,r          (I)  Source positions to compare
 * Function Result : l less than, or equal to r.
 */
Bool STD_CDECL msgSourcePosLessEq( msgSourcePos_t l, msgSourcePos_t r);


/*---------------------------- Exception Handling ----------------------------*/

/*
 * The following defines exception handling:
 */

#define msgTry(__propagate__)                                                           \
           {                                                                            \
              stdThreadContext_t msgEXCT     =  stdGetThreadContext();                  \
              jmp_buf  msgNewContext;                                                   \
              jmp_buf *msgOuterContext       =  msgEXCT->lwrrentContext;                \
              Bool     msgOuterWarningsFound =  msgEXCT->warningsFound;                 \
              Bool     msgOuterErrorsFound   =  msgEXCT->errorsFound;                   \
              Bool     msgPropg              =  __propagate__;                          \
              msgEXCT->lwrrentContext        = &msgNewContext;                          \
              msgEXCT->warningsFound         =  False;                                  \
              msgEXCT->errorsFound           =  False;                                  \
              if ( !setjmp(msgNewContext) ) {                                           \


#define msgOtherwise                                                                    \
                    msgEXCT->lwrrentContext  = msgOuterContext;                         \
                    msgEXCT->warningsFound   = msgOuterWarningsFound                    \
                                               || ((msgPropg)&&msgEXCT->warningsFound); \
                    msgEXCT->errorsFound     = msgOuterErrorsFound                      \
                                               || ((msgPropg)&&msgEXCT->errorsFound);   \
              } else {                                                                  \
                    Bool        msgCaught    = True;                                    \
                    msgMessage  lwrrentError = msgEXCT->raisedException;                \
                    msgEXCT->lwrrentContext  = msgOuterContext;                         \
                    msgEXCT->warningsFound   = msgOuterWarningsFound                    \
                                               || (msgPropg);                           \
                    msgEXCT->errorsFound     = msgOuterErrorsFound                      \
                                               || (msgPropg);                           \
                    {                                                                   \


#define msgOtherwiseSelect                                                              \
                    msgEXCT->lwrrentContext  = msgOuterContext;                         \
                    msgEXCT->warningsFound   = msgOuterWarningsFound                    \
                                               || ((msgPropg)&&msgEXCT->warningsFound); \
                    msgEXCT->errorsFound     = msgOuterErrorsFound                      \
                                               || ((msgPropg)&&msgEXCT->errorsFound);   \
              } else {                                                                  \
                    Bool        msgCaught    = False;                                   \
                    msgMessage  lwrrentError = msgEXCT->raisedException;                \
                    msgEXCT->lwrrentContext  = msgOuterContext;                         \
                    msgEXCT->warningsFound   = msgOuterWarningsFound                    \
                                               || (msgPropg);                           \
                    msgEXCT->errorsFound     = msgOuterErrorsFound                      \
                                               || (msgPropg);                           \
                    if (0) {                                                            \


#define msgCatch(error)                                                                 \
                    } else                                                              \
                    if (lwrrentError == (error)) {                                      \
                        msgCaught= True;      


#define msgDefault                                                                      \
                    } else {                                                            \
                        msgCaught= True;                                                \


#define msgEndTry                                                                       \
                    }                                                                   \
                    if (!msgCaught) {                                                   \
                        msgPropagate();                                                 \
            } } } 


#define msgRaise(exception)                                                             \
    {                                                                                   \
        stdThreadContext_t msgEXCT = stdGetThreadContext();                             \
                                                                                        \
        if (msgEXCT->lwrrentContext == Nil) {                                           \
           stdABORT();                                                                  \
        } else {                                                                        \
           msgEXCT->raisedException= (exception);                                       \
           longjmp( *msgEXCT->lwrrentContext, 1);                                       \
        }                                                                               \
    }


#define msgPropagate()                                                                  \
        msgRaise(lwrrentError)

#define msgEarlyReturn(ret)                                                             \
          {                                                                             \
                    msgEXCT->lwrrentContext  = msgOuterContext;                         \
                    msgEXCT->warningsFound   = msgOuterWarningsFound                    \
                                               || ((msgPropg)&&msgEXCT->warningsFound); \
                    msgEXCT->errorsFound     = msgOuterErrorsFound                      \
                                               || ((msgPropg)&&msgEXCT->errorsFound);   \
                    return ret;                                                         \
          }
 

/*
 * Macro for warning old style source pos struct users:
 */
#define msgCreateSourcePos_phased_out 1


/*----------------------------- Message Reporting ----------------------------*/
/*
 * Support for error log filtering.
 * In case msgAddErrorClassPrefixes is set to True, then
 * all text lines printed due to assertion failures
 * and all text lines printed via msgReport 
 * will start with with an appropriate
 * prefix as defined by the following class prefix
 * macros:
 */
#define msgInfoClassPrefix      "@I@"
#define msgOutputClassPrefix    "@O@"
#define msgWarningClassPrefix   "@W@"
#define msgErrorClassPrefix     "@E@"

/*----------------------------- Getter and Setter Functions ----------------------------*/
/*
 * msgGetToolName() - Obtain the name of the tool using stdlib which is lwrrently being run
 * 
 */
cString STD_CDECL msgGetToolName();

/*
 * msgSetToolName() - set the name of the tool using stdlib which is lwrrently being run
 *                    lMsgToolNane :          Name of the tool which is being lwrrently called upon
 *                                            This string should be caller allocated string and 
 *                                            should also be freed by the caller code
 * 
 */
void STD_CDECL msgSetToolName(cString lMsgToolName);


/*
 * msgGetSuffix() - Obtain the suffix message that is going to be printed along with any error message
 * 
 */
String STD_CDECL msgGetSuffix();

/*
 * msgSetSuffix() - Set the message that is going to be printed suffixed with any error message
 *                  This string should be caller allocated string and 
 *                  should also be freed by the caller code
 * 
 */
void STD_CDECL msgSetSuffix(String lMsgSuffix);


/*
 * msgGetAddErrorClassPrefixes() - Obtain the setting which indicates whether to add any 
 *                                 error class prefix before reporting out error
 * 
 */
Bool STD_CDECL  msgGetAddErrorClassPrefixes();

/*
 * msgSetAddErrorClassPrefixes() - Changes the setting whether to add the error class prefix 
 *                                 before printing out the error or not.
 *                                 If lMsgAddErrorClassPrefixes is -
 *                                   * True  - Error class prefix is added to the error message
 *                                   * False - The error message is printed as is
 *
 */
void STD_CDECL  msgSetAddErrorClassPrefixes(Bool);


/*
 * msgGetDontPrefixErrorLines() - Returns whether every line in a multi-line error is prefixed 
 *                                with error class prefix and tool name
 * 
 */
Bool STD_CDECL   msgGetDontPrefixErrorLines();

/*
 * msgSetDontPrefixErrorLines() - Change the setting of whether everyline in a multi-line error
 *                                is prefixed with error class prefix and tool name.
 *                                If lMsgDontPrefixErrorLines is -
 *                                 * True  - Only the first line gets prefixed
 *                                 * False - Every line gets prefixed
 */
void STD_CDECL   msgSetDontPrefixErrorLines(Bool);

/*
 * msgRestoreDontPrefixErrorLines : Restore the old value of msgDontPrefixErrorLines
 *
 */

void STD_CDECL msgRestoreDontPrefixErrorLines();

/*
 * msgPreserveDontPrefixErrorLines : Preserve the current value of msgDontPrefixErrorLines
 *
 */

void STD_CDECL msgPreserveDontPrefixErrorLines();

/*
 * msgRestoreAddErrorClassPrefixes() - Restore the old value of the addErrorClassPrefix
 *
 */

void STD_CDECL msgRestoreAddErrorClassPrefixes();

/*
 * msgPreserveAddErrorClassPrefixes() - Preserve the current value of the addErrorClassPrefix
 *
 */

void STD_CDECL msgPreserveAddErrorClassPrefixes();


/*
 * msgGetIgnoreWarnings() - Obtain the setting which indicates whether warnings are ignored while reporting
 *
 */
Bool STD_CDECL msgGetIgnoreWarnings();

/*
 * msgSetIgnoreWarnings() - Changes the behaviour of reporting of warnings. If lMsgIgnoreWarning is -
 *                           * True  - Warnings are not reported and are ignored
 *                           * False - Warnings are reported as warnings
 */
void STD_CDECL msgSetIgnoreWarnings(Bool lMsgIgnoreWarnings);


/*
 * msgGetWarnAsError()    - Obtain the setting which indicates whether warnings should be reported as errors
 */
Bool STD_CDECL msgGetWarnAsError();

/*
 * msgSetWarnAsError()    - Changes the behaviour of how warnings are reported.
 *                          If lMsgWarnAsError is -
 *                          * True  - Warnings are treated and reported as errors
 *                          * False - Warnings are treated as warnings
 */
void STD_CDECL msgSetWarnAsError(Bool lMsgWarnAsError);

/*
 * Function        : Report Message with arguments, and exit with status -1
 *                   in case the message's level is Fatal.
 * Parameters      : message (I) name of a message that has been
 *                               specified in local file <prefix>MessageDefs.h
 *                   ...     (I) arguments to be filled in into the representation
 *                               string corresponding with 'message' in <prefix>MessageDefs.h
 * Function Result : -
 */
void STD_CDECL msgReport( msgMessage message, ... );



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
void STD_CDECL msgReportWithPos( msgMessage message, msgSourcePos_t pos, ... );



/*
 * Function        : Report whether messages of level 'Warning' or higher 
 *                   have been reported in the current msgTry context.
 * Parameters      : 
 * Function Result : True iff warnings or higher have been reported.
 */
Bool STD_CDECL msgWarningsFound( void );
#define msgWarningsFound()  stdGetThreadContext()->warningsFound



/*
 * Function        : Report whether messages of level 'Error' or higher 
 *                   have been reported in the current msgTry context. 
 *                   Note that Fatal errors would have resulted 
 *                   in an exception and hence can never be counted
 *                   in the current context.
 * Parameters      : 
 * Function Result : True iff errors or higher have been reported.
 */
Bool STD_CDECL msgErrorsFound( void );
#define msgErrorsFound()  stdGetThreadContext()->errorsFound



/*
 * Function        : Return iff. errors have been reported 
 *                   in the current error context.
 */
void STD_CDECL msgExitOnError( void );
#define msgExitOnError()  if (msgErrorsFound()) { stdEXIT_ERROR(); }



/*
 * Function        : Set warning flag in current msgTry context.
 * Parameters      : warning (I) True or False, to set or clear warning.
 * Function Result : 
 */
void STD_CDECL msgSetWarning( Bool warning );
#define msgSetWarning(warning)  stdGetThreadContext()->warningsFound= (warning);



/*
 * Function        : Set error flag in current msgTry context.
 * Parameters      : error   (I) True or False, to set or clear error.
 * Function Result : 
 */
void STD_CDECL msgSetError( Bool error );
#define msgSetError(error)  stdGetThreadContext()->errorsFound= (error);



/*
 * Function        : Make warnings as errors
 * Parameter       : werror (I) True iff. warnings have to behave as errors
 */
void STD_CDECL msgReportSetWarningAsError( Bool werror );



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
void STD_CDECL msgSwapMessageChannels( Pointer /*stdWriter_t*/ *info, Pointer /*stdWriter_t*/ *warning, Pointer /*stdWriter_t*/ *error );



/*
 * Equality and hash function,
 * for using them in sets and mappings:
 */
uInt STD_CDECL stdSourcePosHash   ( msgSourcePos_t e );
Bool STD_CDECL stdSourcePosEqual  ( msgSourcePos_t e1, msgSourcePos_t e2 );



/*
 * Some late definitions to break include cycles.
 **/
#include "stdLocal.h"

#if     defined(__cplusplus)
}
#endif  /* defined(__cplusplus) */

#endif /* stdComm_INCLUDED */
     





