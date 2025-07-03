/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2013-2015, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ctMessages.h
 *
 *  Description              :
 *     
 */

#include <stdLocal.h>
#include <stdWriter.h>

#ifdef __cplusplus
extern "C" {
#endif

extern FILE *ctMessagesRedirectTo;

/*
 * Function        : Top level init function which calls other ctMessage* initializing routines
 * Parameters      : clientWriter  (IO) Writer to be installed and the current writer would be returned
 * Function Result : 
 */
void ctMessageInit(stdWriter_t* clientWriter);

/*
 * Function        : Top level cleanup function which calls other ctMessage* cleanup routines
 * Parameters      : olderWriter  (IO) Writer will be installed and the current writer would be freed up
 * Function Result : 
 */
void ctMessageExit(stdWriter_t* olderWriter);
   
/*
 * Function        : Create new msgSourcePos_t object
 * Parameters      : fileName     (I) source filename
 *                   lineNo       (I) line number
 *                   sourceStruct (I) Managing msgSourceStruct_t object
 * Function Result : Requested new msgSourcePos_t, other fields set to NULL
 * Note            : Only in JAS path, sourceStruct acts as a manager of  msgSourcePos_t.
 *                   So no explicit freeing of msgSourcePos_t objects are needed in JAS path
 */
msgSourcePos_t ctMsgCreateSourcePos( cString fileName, msgSourceStructure_t* sourceStruct, uInt lineNo );

#ifdef __cplusplus
}
#endif


