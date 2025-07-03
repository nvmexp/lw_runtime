/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "interfaceUtils.h"

/*
 * This mutex is used the compiler and linker interface to exclusively
 *  use the compiler and linker.
 * This is needed as stdlib and other part of PTXAS has global state
 * which makes PTXAS and elfw APIs thread un-safe
 */
class ReentranceMutex
{
public:
    stdMutex_t mutex;
    ReentranceMutex() {
        mutex = stdMutexCreate();
    }
    void enter()  { stdMutexEnter(mutex);}
    void exit()   { stdMutexExit(mutex);}
    ~ReentranceMutex() { stdMutexDelete(mutex); }
};

ReentranceMutex rMutex[NUM_MUTEXES];


void interface_mutex_enter(unsigned int mutexID)
{
    if (mutexID >= NUM_MUTEXES)
        return;

    rMutex[mutexID].enter();
}

void interface_mutex_exit(unsigned int mutexID)
{
    if (mutexID >= NUM_MUTEXES)
        return;

    rMutex[mutexID].exit();
}
