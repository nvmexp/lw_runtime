/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _UNIT_REGOPS_H_
#define _UNIT_REGOPS_H_

#include "lwtypes.h"
#include "devidx.h"

//
//structure to represent which same value to be returned
//& conselwtively how many times
//
struct observer
{   //
    //number of times "value" is returned
    //count = 0 when "value" is returned forever
    //
    LwU32 count;
    // value to return
    LwU32 value;
    //pointer to next node
    struct observer *next;
};

typedef struct observer observer;

//prototype for read call back function
typedef LwU32 ReadCallback(void *);

//prtotype fro write call back function
typedef LwU32 WriteCallback(void *, LwU32);

//
//structure representing a mocked read behaviour applied on
//a particular register
//
struct readRegOp
{
    //address of register
    LwU32 address;
    //value representing how may times a read operation has been performed on this register
    LwU32 timesCalled;
    //
    //callback function to be called to simulate a register
    //if the user specifies a call function to be applied on
    //a register then all other behaviors are supressed and
    //callback function will take priority
    //
    ReadCallback *callback;
    //arguments to be provided to the callback function
    void *callbackArg;
    //pointer to the head of observer list
    observer *observerHead;
    //pointer to the Tail of observer list
    observer *observerTail;
    //pointer to the next readRegOp
    struct readRegOp *next;
};

typedef struct readRegOp readRegOp;

//
//structure containg information for mirroring write.
//
//is a part of "writeRegOp"
//
struct writeMirror
{
    //address of register on which write has to be mirrored
    LwU32 address;
    //pointer to next writeMirror for this writeRegOp
    struct writeMirror *next;
};

typedef struct writeMirror writeMirror;

//
//structure containg written value.
//
//is a part of "writeRegOp"
//
//a Linked list of this structure provides
//information regarding which value was written
//when on this particular register
//
struct writeLogger
{
    //value written
    LwU32 Value;
    //pointer to next writeLogger
    struct writeLogger *next;
};

typedef struct writeLogger writeLogger;

//
//structure representing a mocked write behaviour applied on
//a particular register
//
struct writeRegOp
{
    //address of register
    LwU32 address;
    //buffer to write the value
    LwU32 *regValue;
    //
    //callback function to be called to simulate a register
    //if the user specifies a call function to be applied on
    //a register then all other behaviors are supressed and
    //callback function will take priority
    //
    WriteCallback *callback;
    //arguments to be provided to the callback function
    void *callbackArg;
    //pointer to the head of writelogger list
    writeLogger *writeLoggerHead;
    //pointer to the head of writeMirror List
    writeMirror *writeMirrorHead;
    //pointer to the Tail of writelogger list
    writeLogger *writeLoggerTail;
    //pointer to the Tail of writeMirror List
    writeMirror *writeMirrorTail;
    //pointer to next writeRegOp
    struct writeRegOp *next;
};

typedef struct writeRegOp writeRegOp;

//enum to return the status of install regop operation
enum REGOP_RET_STATUS
{
    REGOP_CALLBACK_APLLIED,
    REGOP_CALLBACK_OVERRIDDEN,
    REGOP_CALLBACK_ERROR,
    REGOP_WILL_RETURN_ALWAYS_APLLIED,
    REGOP_WILL_RETURN_ALWAYS_EROOR,
    REGOP_WILL_RETURN_FOR_COUNT_APLLIED,
    REGOP_WILL_RETURN_FOR_COUNT_ERROR,
    REGOP_WRITE_MIRROR_APPLIED,
    REGOP_WRITE_CALLBACK_APPLIED,
    REGOP_WRITE_CALLBACK_ERROR
};

typedef enum REGOP_RET_STATUS REGOP_RET_STATUS;

//enum to distinguish between a read and a write regOp node
enum REGOP_NODE_TYPE
{
    READ,
    WRITE
};

typedef enum REGOP_NODE_TYPE REGOP_NODE_TYPE;

//
//installs a callback function on a register if the a
//readRegop node is not already present for the given
//register then a new node is created
//
REGOP_RET_STATUS
installRegopReadCallback
(
     LwU32 addr,
     ReadCallback *callbackFun,
     void* callbackArg
 );

#define UTAPI_INSTALL_READ_CALLBACK(addr, callbackfun, callbackarg) installRegopReadCallback\
        (addr, callbackfun, callbackarg)

//
//installs a "will return always" kind of functionality
//on a register if a readRegop node is not already present
//for the given register then a new node is created
//
REGOP_RET_STATUS installRegopReadWillReturnAlways(LwU32 addr, LwU32 value);

#define UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value) \
        installRegopReadWillReturnAlways(addr, value)
//
//installs a "will return for a specific number of times"
//kind of functionality on a register
//if a readRegop node is not already present for the given
//register then a new node is created
//
REGOP_RET_STATUS
installRegopReadWillReturnForCount
(
     LwU32 addr,
     LwU32 value,
     LwU32 count
 );

#define UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value, count) \
        installRegopReadWillReturnForCount(addr, value, count)

//read the value written during nth write
LwU32 UnitReadValueFromNthWrite(LwU32 addr, LwU32 count, LwBool *wasWritten);

#define UTAPI_READ_VALUE_ON_NTH_WRITE(addr, count, waswritten) \
        UnitReadValueFromNthWrite(addr, count, waswritten)

//read the last written value from the register
LwU32 unitReadRegister(LwU32 addr);

#define UTAPI_READ_REGISTER(addr) \
        unitReadRegister(addr)

//
//installs a callback function on a register, if a
//writeRegop node is not already present for the given
//register then a new node is created
//
REGOP_RET_STATUS installRegopWriteCallback
(
     LwU32 addr,
     WriteCallback *callbackFun,
     void* callbackArg
 );

#define UTAPI_INSTALL_WRITE_CALLBACK(addr, callbackfun, callbackarg) \
        installRegopWriteCallback(addr, callbackfun, callbackarg)

//
//Installs "write Mirror" kind of functionality so that write
//to a particular register is mirrored onto another register
//
REGOP_RET_STATUS installRegopWriteMirror(LwU32 addr, LwU32 value);

#define UTAPI_INSTALL_WRITE_MIRROR(addr, value) \
        installRegopWriteMirror(addr, value)

//desroy all lists
void destroyRegopLists();

//forward typedef for OBJGPU
#ifdef DEFINE_OBJGPU
typedef struct OBJGPU OBJGPU;
#endif

//wrapper around unitGpuReadReg, for 32 bit reg
LwU32 unitGpuReadRegister032(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr);

//wrapper around unitGpuWriteReg, for 32 bit reg
void unitGpuWriteRegister032(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr, LwU32 value);

//wrapper around unitGpuReadReg, for 16 bit reg
LwU16 unitGpuReadRegister016(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr);

//wrapper around unitGpuWriteReg, for 16 bit reg
void unitGpuWriteRegister016(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr, LwU16 value);

//wrapper around unitGpuReadReg, for 8 bit reg
LwU8 unitGpuReadRegister008(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr);

//wrapper around unitGpuWriteReg, for 8 bit reg
void unitGpuWriteRegister008(OBJGPU *pGpu, DEVICE_INDEX deviceIndex, LwU32 addr, LwU8 value);

//wrapper around unitGpuReadReg, for pmu-sw
LwU32 unitPmuReadRegister(LwUPtr addr);

//wrapper around unitGpuWriteReg, for pmu-sw
void unitPmuWriteRegister(LwUPtr addr, LwUPtr value);

#endif // _UNIT_REGOPS_H_
