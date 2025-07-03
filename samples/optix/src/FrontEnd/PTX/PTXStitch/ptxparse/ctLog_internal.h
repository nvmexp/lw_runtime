/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ctlog
 *
 */

#ifndef ctLogInternal_INCLUDED
#define ctLogInternal_INCLUDED


#ifdef __cplusplus
extern "C" {
#endif

/*    Function:     getDebugLevel
 *    In      :     void
 *    Out     :     Query function which returns debug level given on command line
*/    
int   getDebugLevel();

/*    Function:     getFileLineInfo 
 *    In      :     void
 *    Out     :     Query function which returns file line information given on command line
*/    
Bool  getFileLineInfo();

/*    Function:     isDebugGroupPresent 
 *    In      :     const char*
 *    Out     :     Query function which returns true if the input group is given on command line
*/    
Bool  isDebugGroupPresent(const char*);

/*    Function:     getOutStream 
 *    In      :     void
 *    Out     :     Query function which returns stream ID to which messages will be logged.
*/    
FILE* getOutStream();


/*    Function:     isLoggerInitialized
 *    In      :     void
 *    Out     :     Query function which returns if the logger is initialized or not.
*/    
Bool isLoggerInitialized();

#ifdef __cplusplus
}
#endif 


#endif      // File inclusion guard ends.

