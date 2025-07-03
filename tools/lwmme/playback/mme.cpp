#define _CRT_SELWRE_NO_DEPRECATE 1
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include "mme.h"

#define TRACE_PRINT_BUF_LEN 256

void MMEProgram::startRun(unsigned int *stateInput, int stateInputSize, unsigned int *methodTriggerState, int methodTriggerStateSize) {
    warningCat[0] = warningMsg[0] = '\0';

    isError = false;

    for(int i=0; i<NUM_METHODS; i++) state[i] = 0;
    stateWritten.reset();

    // Init the state store
    if (stateInputSize & 1) { setError("State data not in method/data pairs"); }
    for(int i=0; i<(int)stateInputSize; i+=2) SetState(stateInput[i], stateInput[i+1]);
    // Check the trigger state
    if ((methodTriggerStateSize % 3) != 0) { setError("State trigger data not in trigger/method/data triples"); }

    m_methodTriggerState = methodTriggerState;
    m_methodTriggerStateSize = methodTriggerStateSize;
}

void MMEProgram::tprintf(const char *fmt, ... ) {
    if (!trace) return;

    char s[TRACE_PRINT_BUF_LEN];
    int len;
    va_list args;
    va_start( args, fmt );
    len = vsnprintf( s, TRACE_PRINT_BUF_LEN, fmt, args );
    va_end( args );

    if (len >= TRACE_PRINT_BUF_LEN) {
        mmeTrace( "ERROR next print overflows print buffer and is truncated" );
    }

    mmeTrace( s );
}

void MMEProgram::setError(const char *errorString) {
    // Only report the first error
    if (isError) return;

    isError = true;
    strcpy(error, "Error: ");
    strcat(error, errorString);
}

void MMEProgram::setWarning(const char *cat, const char *msg) {
    strcpy(warningCat, cat);
    strcpy(warningMsg, msg);
}

unsigned int MMEProgram::GetState(unsigned int addr) {
    if ((addr >= NUM_METHODS) ||
        !stateWritten[addr]) {
        char buf[256];
        sprintf(buf, "Tried to access undefined or out-of-range method at 0x%08x (%d)\n", addr, addr);
        setError(buf);
        return 0;
    }

    return state[addr];
}

void MMEProgram::SetState(unsigned int addr, unsigned int data) {
    state[addr] = data;
    stateWritten.set(addr);
}

void MMEProgram::Release(void *callbackData, unsigned int method, unsigned int data) {
    cb(callbackData, method, data);
    SetState(method, data);
    if (m_methodTriggerState) {
        for (int mt=0; mt<m_methodTriggerStateSize; mt += 3) {
            if (method == m_methodTriggerState[mt]) {
                SetState(m_methodTriggerState[mt+1], m_methodTriggerState[mt+2]);
            }
        }
    }
}
