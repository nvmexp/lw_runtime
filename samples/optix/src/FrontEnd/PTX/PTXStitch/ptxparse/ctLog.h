/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2018, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ctLog
 *
 *  Description              :
 *
 */


#include <stdSet.h>
#include "ctLog_internal.h"
#include "g_lwconfig.h"

#ifndef ctLog_INCLUDED
#define ctLog_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

struct ctLogInfo
{
    stdSet_t     debugGroups;                 // Store debug groups input on command line.
    int          debugLevel;                  // Stores debug level input on command line.
    Bool         fileLineInfoEnabled;         // Does user want file and line info in log?
};

/*    Function:     ctLogInitialize
 *    In      :     struct ctLogInfo*
 *    Out     :     void
 *    Comment :     Initializer function for the logger.
 *                  client must call this function before starting to log.
 *                  If the function is not called then in debug release
 *                  all successive framweork's APIs will assert and in release
 *                  version APIs won't perform any action.
*/
void ctLogInitialize(struct ctLogInfo *o);

/*    Function:     ctLogSetOutStream
 *    In      :     FILE*
 *    Out     :     void
 *    Comment :     Set o/p stream to which all log messages wil be emitted.
*/
void ctLogSetOutStream(FILE *f);

/*    Function:     ctLogResetOutStream
 *    In      :     void
 *    Out     :     void
 *    Comment :     Reset o/p stream to stdout
*/
void ctLogResetOutStream();

/*    Function:     ctLogFinish
 *    In      :     void
 *    Out     :     void
 *    Comment :     Finsher Function for the logger.
 *                  This function will clear all internal states of
 *                  the logger as well as the ctLogInfo* which is passed to ctLogInitialize.
*/
void ctLogFinish();

// In release builds (of even non-release branch), there are checks for some guardwords
// So avoid using filenames in ALL release build flavours
#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL) && !defined(RELEASE)
    #define __ptx_FILE__ __FILE__
#else
    #define __ptx_FILE__ "<src>"
#endif

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define CT_DEBUG_BEGIN(GROUP_TITLE)   \
        do  \
        {   \
            if (isLoggerInitialized())  \
            { \
                if (isDebugGroupPresent(GROUP_TITLE)) \
                {   \
                    fprintf(getOutStream(), "\n***** BEGIN PHASE: %s\n", GROUP_TITLE);\
                    if (getFileLineInfo()) \
                    fprintf(getOutStream(),  "***** AT LOCATION: %s, %d\n", __ptx_FILE__, __LINE__);  \
                }  \
            } \
        }while(0)
#else
    #define CT_DEBUG_BEGIN(GROUP_TITLE)
#endif

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define CT_DEBUG_END(GROUP_TITLE)   \
        do  \
        {   \
            if (isLoggerInitialized())  \
            { \
                if (isDebugGroupPresent(GROUP_TITLE)) \
                {   \
                    fprintf(getOutStream(), "\n***** END PHASE: %s\n", GROUP_TITLE);\
                    if (getFileLineInfo()) \
                    fprintf(getOutStream(),  "***** AT LOCATION: %s, %d\n", __ptx_FILE__, __LINE__);  \
                }  \
            } \
        }while(0)
#else
    #define CT_DEBUG_END(GROUP_TITLE)
#endif

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define CT_DEBUG_PRINT(GROUP_TITLE, DEBUG_LEVEL, DEBUG_MESSAGE, ...)   \
        do  \
        {   \
            if (isLoggerInitialized())  \
            { \
                if ((getDebugLevel() == -1 || DEBUG_LEVEL <= getDebugLevel())  \
                    && (isDebugGroupPresent(GROUP_TITLE) )) \
                {   \
                    fprintf(getOutStream(), DEBUG_MESSAGE, ##__VA_ARGS__);\
                }  \
            } \
        } while(0)
#else
    #define CT_DEBUG_PRINT(GROUP_TITLE, DEBUG_LEVEL, DEBUG_MESSAGE, ...)
#endif

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define CT_DEBUG_MSG(GROUP_TITLE, DEBUG_LEVEL, DEBUG_MESSAGE, ...)   \
        do  \
        {   \
            if (isLoggerInitialized())  \
            { \
                if ((getDebugLevel() == -1 || DEBUG_LEVEL <= getDebugLevel())  \
                    && (isDebugGroupPresent(GROUP_TITLE) )) \
                {   \
                    fprintf(getOutStream(), "***** %s :: ", GROUP_TITLE);\
                    if (getFileLineInfo()) \
                        fprintf(getOutStream(), "%s:%d: ", __ptx_FILE__, __LINE__);  \
                    fprintf(getOutStream(), DEBUG_MESSAGE, ##__VA_ARGS__);\
                }  \
            } \
        }while(0)
#else
    #define CT_DEBUG_MSG(GROUP_TITLE, DEBUG_LEVEL, DEBUG_MESSAGE, ...)
#endif

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define CT_DEBUG_DO(GROUP_TITLE, DEBUG_LEVEL, DEBUG_ACTION) \
        do  \
        {   \
            if (isLoggerInitialized())\
            { \
                if ((getDebugLevel() == -1 || DEBUG_LEVEL <= getDebugLevel())  \
                    && (isDebugGroupPresent(GROUP_TITLE))) \
                    DEBUG_ACTION \
            } \
        }while(0)
#else
    #define CT_DEBUG_DO(GROUP_TITLE, DEBUG_LEVEL, DEBUG_ACTION)
#endif

#ifdef __cplusplus
}
#endif


#endif              // File inclusion guard ends.
