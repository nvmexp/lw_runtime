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
 *  Module name              : ctLog
 *
 *  Description              :
 * 
 */


#include <stdio.h>
#include "ctLog.h"

// ************ Framework's private APIs ****************

struct logState {
    struct ctLogInfo *opts;
    FILE *logOutStream;
};

typedef struct logState* logState;
static logState l = NULL;
#define VALIDATE_LOG_STATE() stdASSERT(l, ("Logger not initialized. Did you forget to call ctLogInitialize()?"))

int getDebugLevel()
{
    VALIDATE_LOG_STATE();
    return l->opts->debugLevel;
}

Bool getFileLineInfo()
{
    VALIDATE_LOG_STATE();
    return l->opts->fileLineInfoEnabled;
}

Bool isDebugGroupPresent(const char *c)
{
    // If debugGroups is set to NULL by client then 
    // he wants all groups to be in log.

    VALIDATE_LOG_STATE();
    if (!l->opts->debugGroups) {
        return True; 
    } else {
        return setContains(l->opts->debugGroups, (Pointer)c);        
    }
}

FILE* getOutStream()
{
    VALIDATE_LOG_STATE();
    return l->logOutStream;
}

Bool isLoggerInitialized()
{
    return l == NULL ? False : True;
}
 
// ************ Framework's public APIs ****************

void ctLogInitialize(struct ctLogInfo *o)
{
    if (!l) {
        l                      = (logState)stdMALLOC(sizeof(struct logState));
        l->logOutStream        = stdout;
        l->opts                = o;
    } else {
        stdASSERT(0, ("Logger already initialized. First clear the exisiting logger to create a new one\n"));
    }
}

void ctLogSetOutStream(FILE *f)
{
    VALIDATE_LOG_STATE();
    if (isLoggerInitialized())
        l->logOutStream = f;
}

void ctLogResetOutStream()
{
    VALIDATE_LOG_STATE();
    if (isLoggerInitialized())
        l->logOutStream = stdout;
}

void ctLogFinish()
{
    if (isLoggerInitialized()) {
        stdFREE(l->opts);
    
        // Clear file pointer only if it is not stdout and stderr.
        if (l->logOutStream != stdout && l->logOutStream != stderr)
            fclose(l->logOutStream);

        stdFREE(l);
        
        l = NULL;
    }
}
