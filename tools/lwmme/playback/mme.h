#ifndef _MME_H
#define _MME_H 1

#include <assert.h>
#include "common/mmebitset.h"

// Conservative estimate
#define NUM_METHODS 0x10000

extern void mmeTrace(const char *msg);

static inline unsigned int GetBits(unsigned int MSB, unsigned int LSB, unsigned int source) {
    return (source >> LSB) & ((1 << (MSB - LSB + 1)) - 1);
}

template <class T>
static void SetBits(T &dest, unsigned int MSB, unsigned int LSB, unsigned int source) {
    T temp = source;
    T mask = (1 << (MSB - LSB + 1)) - 1;

    temp &= mask;
    temp <<= LSB;

    dest &= ~(mask << LSB);
    dest |= temp;
}

class MMEProgram {
public:
    typedef void (*Callback)(void*, unsigned int, unsigned int);
    MMEProgram(Callback callback, bool t=false) :
        isError(false),
        m_methodTriggerState(nullptr),
        m_methodTriggerStateSize(0),
        cb(callback),
        trace(t)
    {
    }

    virtual ~MMEProgram() {
    }

    virtual int run(void *callbackData,
        unsigned int *data, int dataSize,
        unsigned int *stateInput, int stateInputSize,
        unsigned int *methodTriggerState, int methodTriggerStateSize,
        unsigned int *dataRamState, int dataRamStateSize,
        unsigned int *memoryState, int memoryStateSize) = 0;

    char * getError() {return &error[0];}
    char * getWarningCat() {return &warningCat[0];}
    char * getWarningMsg() {return &warningMsg[0];}

    virtual int numInsts() const = 0;

    void setError(const char *);
    void setWarning(const char *, const char *);
    void tprintf(const char *fmt, ... );

protected:
    bool isError;

    void startRun(unsigned int *stateInput, int stateInputSize, unsigned int *methodTriggerState, int methodTriggerStateSize);

    unsigned int GetState(unsigned int addr);
    void SetState(unsigned int addr, unsigned int data);
    void Release(void *callbackData, unsigned int method, unsigned int data);

private:
    unsigned int state[NUM_METHODS];
    mme::bitset<NUM_METHODS> stateWritten;

    unsigned int *m_methodTriggerState;
    int m_methodTriggerStateSize;

    Callback cb;

    bool trace;

    char error[256];
    char warningCat[256];
    char warningMsg[256];

private:
    // Uncopyable
    MMEProgram(MMEProgram &other);
    MMEProgram& operator=(const MMEProgram & );
};

#endif
