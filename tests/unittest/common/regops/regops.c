/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdlib.h>
#define DEFINE_OBJGPU
#include "regops.h"
#include "utility.h"

//
//pointer to the head of list of read register
//operations applied on registers
//
static readRegOp *readRegopHead;

//
//pointer to the head of list of write register
//operations applied on registers
//
static writeRegOp *writeRegopHead;

//
//pointer to the Tail of list of read register
//operations applied on registers
//
static readRegOp *readRegopTail;

//
//pointer to the Tail of list of write register
//operations applied on registers
//
static writeRegOp *writeRegopTail;

/*!
 * @brief creates a new observer node
 *
 * @param[in]      count    "count" to initialize with
 * @param[in]      value    "value" to initialize with
 *
 * @return         address of the createed observer node
 */
static observer*
createNewObserverNode
(
    LwU32 count,
    LwU32 value
)
{
    observer *observerNode = (observer*)malloc(sizeof(observer));

    observerNode->count = count;
    observerNode->value = value;
    observerNode->next = NULL;

    return observerNode;
}

/*!
 * @brief free observer node list
 *
 * @param[in]      head     pointer to the head of the observer list
 *
 */
static void
freeObserverList
(
    observer *head
)
{
    observer* iter = head;
    observer* temp;

    UNIT_ASSERT( head != NULL);

    while (iter)
    {
        temp = iter->next;
        free(iter);
        iter = temp;
    }
}

/*!
 * @brief creates a new writeMirror node
 *
 * @param[in]      value    "Register address" to initialize with
 *
 * @return         address of the created observer node
 */
static writeMirror*
createNewWriteMirrorNode
(
    LwU32 value
)
{
        writeMirror *node = (writeMirror*)malloc(sizeof(writeMirror));
        node->address = value;
        node->next = NULL;
        return node;
}

/*!
 * @brief free the writeMirror node list
 *
 * @param[in]      head    pointer to the head of the writeMirror list
 *
 */
static void
freeWriteMirrorList
(
    writeMirror *head
)
{
    writeMirror* iter = head;
    writeMirror* temp;

    UNIT_ASSERT( head != NULL);

    while (iter)
    {
        temp = iter->next;
        free(iter);
        iter = temp;
    }
}

/*!
 * @brief creates a new node of type readRegOp
 *
 * @param[in]      addr        "Register address" associated with node
 * @param[in]      callback    callback function used to mock register
 * @param[in]      callbackArg argument for the callback function
 *
 * @return         address of the created node
 */
static readRegOp*
createNewReadRegop
(
    LwU32 addr,
    ReadCallback *callback,
    void *callbackArg,
    observer *observerNode
)
{
    readRegOp* node = (readRegOp*)malloc(sizeof(readRegOp));

    node->address = addr;
    node->callback = callback;
    node->callbackArg = callbackArg;
    node->observerHead = observerNode;
    node->observerTail = observerNode;
    node->timesCalled = 0;
    node->next = NULL;

    return node;
}

/*!
 * @brief free readRegOp list
 *
 * @param[in]      head        pointer to the head of the readRegop list
 *
 */
static void
freeReadRegopList
(
    readRegOp *head
)
{
    readRegOp *iter = head;
    readRegOp *temp;

    UNIT_ASSERT( head != NULL);

    while (iter)
    {
        temp = iter->next;

        if (iter->observerHead)
            freeObserverList(iter->observerHead);

        free(iter);
        iter = temp;
    }
}

/*!
 * @brief creates a new node of type writeRegOp
 *
 * @param[in]      addr            "Register address" associated with node
 * @param[in]      callback        callback function used to mock register
 * @param[in]      callbackArg     argument for the callback function
 * @param[in]      writeMirrorNode pinter to teh writeMirror used for init
 *
 * @return         address of the created node
 */
static writeRegOp*
createNewWriteRegop
(
    LwU32 addr,
    WriteCallback *callback,
    void *callbackArgs,
    writeMirror *writeMirrorNode
)
{
    writeRegOp* node = (writeRegOp*)malloc(sizeof(writeRegOp));

    node->address = addr;
    node->callback = callback;
    node->callbackArg = callbackArgs;
    node->writeLoggerHead = NULL;
    node->writeLoggerTail = NULL;
    node->writeMirrorHead = writeMirrorNode;
    node->writeMirrorTail = writeMirrorNode;
    node->regValue = NULL;
    node->next = NULL;

    return node;
}

/*!
 * @brief free the writelogger list
 *
 * @param[in]      head        pointer to the head of the writeLogger list
 *
 */
static void
freeWriteLoggerList
(
    writeLogger *head
)
{
    writeLogger *iter = head;
    writeLogger *temp;

    while (iter)
    {
        temp = iter->next;
        free(iter);
        iter = temp;
    }
}

/*!
 * @brief create a new logger node and insert it after the current tail
 *
 * @param[in]      node        address of the node in which logger has to
 *                             be inserted
 * @param[in]      value       value written
 *
 * @return         address of the created node
 */
static writeLogger*
createAndAppendLoggerNode
(
    writeRegOp *node, LwU32 value
)
{
    writeLogger *loggerNode;

    loggerNode = (writeLogger *)malloc(sizeof(writeLogger));

    loggerNode->Value = value;
    loggerNode->next = NULL;

    // no head/tail, create a new node and make this head/tail
    if ( !node->writeLoggerTail)
    {
        node->writeLoggerHead = loggerNode;
        node->writeLoggerTail = loggerNode;
    }

    //append the node
    else
    {
        (node->writeLoggerTail->next) = loggerNode;
        (node->writeLoggerTail) = loggerNode;
    }

    return loggerNode;
}

/*!
 * @brief free writeRegOp list
 *
 * @param[in]      head        pointer to the head of the writeRegop list
 *
 */
static void
freeWriteRegopList
(
    writeRegOp *head
)
{
    writeRegOp *iter = head;
    writeRegOp *temp;

    UNIT_ASSERT( head != NULL);

    while (iter)
    {
        temp = iter->next;

        if(iter->regValue)
            free(iter->regValue);
        if (iter->writeMirrorHead)
            freeWriteMirrorList(iter->writeMirrorHead);

        if (iter->writeLoggerHead)
            freeWriteLoggerList(iter->writeLoggerHead);

        free(iter);
        iter = temp;
    }

}

/*!
 * @brief check if there is already a REGOP_WILL_RETURN_ALWAYS_APLLIED
 *        on this register
 * @param[in]      observerHead    ptr to head of the observer list
 *
 * @return         TRUE            if observer with count 0 present
 *                 FALSE           otherwise
 */
static LwBool
willReturn_Present
(
    observer *observerHead
)
{
    observer *iter = observerHead;
    while (iter)
    {
        if(iter->count == 0)
            return TRUE;

        iter = iter->next;
    }

    return FALSE;
}

/*!
 * @brief return the addr of a regop node for a given address from the list
 *
 * @param[in]      head    ptr to head of regop node list(read/write)
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      type    READ/WRITE
 *
 * @return         address of regop node
 */
static void *
findRegopNode
(
    void *head,
    LwU32 addr,
    REGOP_NODE_TYPE type
)
{
    readRegOp* readHead = NULL;
    writeRegOp* writeHead = NULL;

    readRegOp* readIter = NULL;
    writeRegOp* writeIter = NULL;

    if (type == READ)
    {
        readHead = (readRegOp*)head;
        readIter = readHead;

        while (readIter != NULL)
        {
            if (readIter->address == addr)
            {
                return readIter;
            }
            readIter = readIter->next;
        }

        return NULL;
    }

    else
    {
        writeHead = (writeRegOp*)head;
        writeIter = writeHead;

        while (writeIter != NULL)
        {
            if (writeIter->address == addr)
            {
                return writeIter;
            }
            writeIter = writeIter->next;
        }

        return NULL;
    }

}

/*!
 * @brief installs a callback function on a register, if a
 *        readRegop node is not already present for the given
 *        register then a new node is created
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      callbackFun    function to model the register
 *
 * @param[in]      callbackArg    argument to the above function
 *
 * @return         REGOP_CALLBACK_APLLIED in case of success
 *                 REGOP_CALLBACK_ERROR   otherwise
 */
REGOP_RET_STATUS
installRegopReadCallback
(
    LwU32 addr,
    ReadCallback *callbackFun,
    void* callbackArg
 )
{
    readRegOp *iter = readRegopHead;

    //If this is the first register for which a regOp is applied
    //then initialize the readRegopHead with readRegOp for this register
    //as it would be NULL
    if (readRegopHead == NULL)
    {
        iter = createNewReadRegop(addr, callbackFun, callbackArg, NULL);
        readRegopHead = iter;
        readRegopTail = iter;

        return REGOP_CALLBACK_APLLIED;
    }

    //check if there is already a readRegOp present on this register
    iter = findRegopNode(readRegopHead, addr, READ);

    //readRegOp present on this register update the elements
    if (iter)
    {
        // a user modelled register may not have any other
        // infra specific behavior
        return REGOP_CALLBACK_ERROR;
    }

    //readRegOp not present on this reg, create a new node
    else
    {
        readRegopTail->next = createNewReadRegop(addr, callbackFun, callbackArg, NULL);
        readRegopTail = readRegopTail->next;

        return REGOP_CALLBACK_APLLIED;
    }
}

/*!
 * @brief installs a "will return always" kind of functionality
 *        on a register if a readRegop node is not already present
 *        for the given register then a new node is created
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      value   value to be returned
 *
 * @return         REGOP_WILL_RETURN_ALWAYS_APLLIED in case of success
 *                 REGOP_WILL_RETURN_ALWAYS_ERROR   otherwise
 */
REGOP_RET_STATUS
installRegopReadWillReturnAlways
(
    LwU32 addr,
    LwU32 value
 )
{

    readRegOp *iter = readRegopHead;
    observer *observerNode = NULL;
    observer *observerIter = NULL;

    //If this is the first register for which a regOp is applied
    //then initialize the readRegopHead with readRegOp for this register
    //as it would be NULL
    if (readRegopHead == NULL)
    {
        //
        //create a new observer node with count 0 i.e. always
        //return this particluar value
        //
        observerNode = createNewObserverNode(0, value);

        iter = createNewReadRegop(addr, NULL, NULL, observerNode);
        readRegopHead = iter;
        readRegopTail = iter;

        return REGOP_WILL_RETURN_ALWAYS_APLLIED;
    }

    //check if there is already a readRegOp present on this register
    iter = findRegopNode(readRegopHead, addr, READ);

    //readRegOp present on this register
    if (iter)
    {
        if ( willReturn_Present(iter->observerHead) )
        {
            //
            //there can be only one WILL_RETURN_ALWAYS
            //for a given register, Fail observer installation
            //
            return REGOP_WILL_RETURN_ALWAYS_EROOR;
        }

        else
        {
            //get the last observer node
            observerIter = iter->observerTail;
            //
            //create a new observer node with count 0 i.e. always
            //return this particluar value
            //
            observerNode = createNewObserverNode(0, value);
            observerIter->next = observerNode;
            iter->observerTail = observerNode;

            return REGOP_WILL_RETURN_ALWAYS_APLLIED;
        }

    }
    //readRegOp not present on this reg, create a new node
    else
    {
        //
        //create a new observer node with count 0 i.e. always
        //return this particluar value
        //
        observerNode = createNewObserverNode(0, value);

        //create a new readRegOp node
        readRegopTail->next = createNewReadRegop(addr, NULL, NULL, observerNode);
        readRegopTail = readRegopTail->next;

        return REGOP_WILL_RETURN_ALWAYS_APLLIED;
    }
}

/*!
 * @brief installs a "will return for a specific number of times"
 *        kind of functionality on a register
 *        if a readRegop node is not already present for the given
 *        register then a new node is created
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      value   value to be returned
 *
 * @param[in]      count   number of times the value has to be returned
 *
 * @return         REGOP_WILL_RETURN_FOR_COUNT_APLLIED in case of success
 *                 REGOP_WILL_RETURN_FOR_COUNT_ERROR   otherwise
 */
REGOP_RET_STATUS
installRegopReadWillReturnForCount
(
    LwU32 addr,
    LwU32 value,
    LwU32 count
 )
{
    readRegOp *iter = readRegopHead;
    observer *observerNode = NULL;
    observer *observerIter = NULL;

    if ( count==0 )
    {
        //count = 0, reserved for will return always
        return REGOP_WILL_RETURN_FOR_COUNT_ERROR;
    }

    //If this is the first register for which a regOp is applied
    //then initialize the readRegopHead with readRegOp for this register
    //as it would be NULL
    if (readRegopHead == NULL)
    {

        //create a new observer node with given count & value
        observerNode = createNewObserverNode(count, value);

        iter = createNewReadRegop(addr, NULL, NULL, observerNode);
        readRegopHead = iter;
        readRegopTail = iter;

        return REGOP_WILL_RETURN_FOR_COUNT_APLLIED;
    }

    //check if there is already a readRegOp present on this register
    iter = findRegopNode(readRegopHead, addr, READ);

    //readRegOp present on this register
    if (iter)
    {
        if ( willReturn_Present(iter->observerHead) || iter->callback )
        {
            //
            //if there is a callback present or a will
            //return always present then fail
            //
            return REGOP_WILL_RETURN_FOR_COUNT_ERROR;
        }

        else
        {
            //get the last observer node
            observerIter = iter->observerTail;

            //create a new observer node with given count & value
            observerNode = createNewObserverNode(count, value);
            observerIter->next = observerNode;
            iter->observerTail = observerNode;

            return REGOP_WILL_RETURN_FOR_COUNT_APLLIED;
        }

    }
    //readRegOp not present on this reg, create a new node
    else
    {
        //create a new observer node with given count & value
        observerNode = createNewObserverNode(count, value);

        //create a new readRegOp node
        readRegopTail->next = createNewReadRegop(addr, NULL, NULL, observerNode);
        readRegopTail = readRegopTail->next;

        return REGOP_WILL_RETURN_FOR_COUNT_APLLIED;
    }
}

/*!
 * @brief Installs "write Mirror" kind of functionality so that write
 *        to a particular register is mirrored onto another register
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      value   address of the reg on which write has to be mirrored
 *
 * @return         REGOP_WRITE_MIRROR_APPLIED in case of success
 *                 REGOP_WRITE_MIRROR_ERROR   otherwise
 */
REGOP_RET_STATUS
installRegopWriteMirror
(
    LwU32 addr,
    LwU32 value
 )
{
    writeRegOp *iter;
    writeMirror *node;

    //
    //If this is the first register for which a regOp is applied
    //then initialize the writeRegopHead with writeRegOp for this register
    //as it would be NULL
    //
    if (writeRegopHead == NULL)
    {
        node = createNewWriteMirrorNode(value);
        iter = createNewWriteRegop(addr, NULL, NULL, node);

        writeRegopHead = iter;
        writeRegopTail = iter;

        return REGOP_WRITE_MIRROR_APPLIED;
    }

    //check if there is already a writeRegOp present on this register
    iter = findRegopNode(writeRegopHead, addr, WRITE);

    //writeRegOp present on this register
    if (iter)
    {
        node = createNewWriteMirrorNode(value);

        //
        //if there are other mirrors in this node
        // then append this node at the end
        //
        if (iter->writeMirrorTail)
        {
            iter->writeMirrorTail->next = node;
            iter->writeMirrorTail = node;

            return REGOP_WRITE_MIRROR_APPLIED;
        }

        //
        //there are no mirrors present
        //update the head
        //
        else
        {
            iter->writeMirrorHead = node;
            iter->writeMirrorTail = node;

            return REGOP_WRITE_MIRROR_APPLIED;
        }

    }

    //no writeRegOp present create a new one
    else
    {
        node = createNewWriteMirrorNode(value);
        iter = createNewWriteRegop(addr, NULL, NULL, node);

        writeRegopTail->next = iter;
        writeRegopTail = iter;

        return REGOP_WRITE_MIRROR_APPLIED;
    }
}

/*!
 * @brief installs a callback function on a register, if a
 *        writeRegop node is not already present for the given
 *        register then a new node is created
 *
 * @param[in]      addr    address of register
 *
 * @param[in]      callbackFun    function to model the register
 *
 * @param[in]      callbackArg    argument to the above function
 *
 * @return         REGOP_WRITE_CALLBACK_APPLIED in case of success
 *                 REGOP_WRITE_CALLBACK_ERROR   otherwise
 */
REGOP_RET_STATUS
installRegopWriteCallback
(
    LwU32 addr,
    WriteCallback *callbackFun,
    void* callbackArg
 )
{
    writeRegOp *iter;

    //
    //If this is the first register for which a regOp is applied
    //then initialize the writeRegopHead with writeRegOp for this register
    //as it would be NULL
    //
    if (writeRegopHead == NULL)
    {
        iter = createNewWriteRegop(addr, callbackFun, callbackArg, NULL);

        writeRegopHead = iter;
        writeRegopTail = iter;

        return REGOP_WRITE_CALLBACK_APPLIED;
    }

    //check if there is already a writeRegOp present on this register
    iter = findRegopNode(writeRegopHead, addr, WRITE);

    //writeRegOp present on this register
    if (iter)
    {
        //if a callback already present then return error
        if (iter->callback)
            return REGOP_WRITE_CALLBACK_ERROR;

        //else, update teh callback
        else
        {
            iter->callback = callbackFun;
            iter->callbackArg = callbackArg;

            return REGOP_WRITE_CALLBACK_APPLIED;
        }
    }

    //no writeRegOp present, create a new one
    else
    {
        iter = createNewWriteRegop(addr, callbackFun, callbackArg, NULL);

        writeRegopTail->next = iter;
        writeRegopTail = iter;

        return REGOP_WRITE_CALLBACK_APPLIED;
    }
}

/*!
 * @brief read the last written value from the register
 *
 * @param[in]      addr        address of register
 *
 * @param[in,out]  wasWritten  pointer to LwBool var
 *                             TRUE if reg was ever Written
 *                             FALSE otherwie
 *
 * @return         value read
 */
static LwU32
regopReadRegister
(
    LwU32 addr,
    LwBool *wasWritten
)
{
    writeRegOp *iter;

    //find the writeRegOp associated with this register
    iter = findRegopNode(writeRegopHead, addr, WRITE);

    //writeRegOp present, return regValue
    if (iter)
    {
        //register was never written
        if (!iter->regValue)
        {
            *wasWritten = FALSE;
            return 0xACDCACDC;
        }

        else
        {
            *wasWritten = TRUE;
            return *(iter->regValue);
        }
    }

    //
    //no writeRegOp present, register was not written
    //return default value
    //
    else
    {
        *wasWritten = FALSE;
        return 0xACDCACDC;
    }
}

/*!
 * @brief wrapper around regopReadRegister
 *
 * @param[in]      addr        address of register
 *
 * @return         value read
 */
//
LwU32
unitReadRegister
(
    LwU32 addr
)
{
    LwBool wasWritten;
    LwU32 value;

    value = regopReadRegister(addr, &wasWritten);

    UNIT_ASSERT( wasWritten != FALSE);

    return value;
}

/*!
 * @brief read the value written during nth write
 *
 * @param[in]      addr        address of register
 *
  * @param[in]     count       the value of N
 *
 * @param[in,out]  wasWritten  pointer to LwBool var
 *                             TRUE if reg was ever Written
 *                             FALSE otherwie
 *
 * @return         value read
 */
LwU32
UnitReadValueFromNthWrite
(
    LwU32 addr,
    LwU32 count,
    LwBool *wasWritten
)
{
    writeRegOp *iter;
    LwU32 logCount = 1;
    writeLogger *loggerNode;

    UNIT_ASSERT(wasWritten);

    //find the writeRegOp associated with this register
    iter = findRegopNode(writeRegopHead, addr, WRITE);

    //writeRegOp present, read from log
    if (iter)
    {
        //no log present
        if ( !iter->writeLoggerHead )
        {
            *wasWritten = FALSE;
            return 0xACDCACDC;
        }

        else
        {
            loggerNode = iter->writeLoggerHead;

            //find the value at Nth node of writeLogger
            while (loggerNode)
            {
                if (logCount == count)
                    break;

                logCount++;
                loggerNode = loggerNode->next;
            }

            //correct node found
            if (logCount == count)
            {
                *wasWritten = TRUE;
                return loggerNode->Value;
            }

            //no node found at nth position
            else
            {
                *wasWritten = FALSE;
                return 0xACDCACDC;
            }
        }
    }

    //
    //no writeRegOp present, register was not written
    //return default value
    //
    else
    {
        *wasWritten = FALSE;
        return 0xACDCACDC;
    }
}

/*!
 * @brief desroy all lists
 *
 */
void
    destroyRegopLists()
{
    if (readRegopHead)
    {
        freeReadRegopList(readRegopHead);
        readRegopHead = NULL;
    }

    if (writeRegopHead)
    {
        freeWriteRegopList(writeRegopHead);
        writeRegopHead = NULL;
    }
}

/*!
 * @brief Unit test Infra gpu write function
 *
 * @param[in]      addr        address of register
 *
  * @param[in]     value       value to be written
 *
 */
static void
unitGpuWriteReg
(
    LwU32 addr,
    LwU32 value
 )
{
    writeRegOp *iter;
    writeMirror *node;
    writeLogger *loggerNode;
    LwU32 modelledValue;

    //
    //If writeRegop list is empty
    //then initialize the writeRegopHead with writeRegOp
    // for this register as it would be NULL
    //
    if (writeRegopHead == NULL)
    {
        iter = createNewWriteRegop(addr, NULL, NULL, NULL);

        writeRegopHead = iter;
        writeRegopTail = iter;

        iter->regValue = (LwU32*)malloc(sizeof(LwU32));
        *(iter->regValue) = value;

        loggerNode = createAndAppendLoggerNode(iter, value);
    }

    //writeRegOp list present
    else
    {
        //check if there is writeRegOp node for this reg
        iter = findRegopNode(writeRegopHead, addr, WRITE);

        //writeRegOp node present
        if (iter)
        {
            //if there are any mirrors, then handle them
            if (iter->writeMirrorHead)
            {
                node = iter->writeMirrorHead;
                while (node)
                {
                    unitGpuWriteReg(node->address, value);
                    node = node->next;
                }
            }

            //if there is a callback, then call the callback
            if (iter->callback)
            {
                modelledValue = iter->callback(iter->callbackArg, value);

                if (!iter->regValue)
                    iter->regValue = (LwU32*)malloc(sizeof(LwU32));

                *(iter->regValue) = modelledValue;

                loggerNode = createAndAppendLoggerNode(iter, modelledValue);
            }

            else
            {
                if (!iter->regValue)
                    iter->regValue = (LwU32*)malloc(sizeof(LwU32));

                *(iter->regValue) = value;

                loggerNode = createAndAppendLoggerNode(iter, value);
            }
        }

        //
        //no writeRegop present for this register
        //this is the firs write for this reg, create a node
        //
        else
        {
        iter = createNewWriteRegop(addr, NULL, NULL, NULL);

        writeRegopTail->next = iter;
        writeRegopTail = iter;

        if (!iter->regValue)
            iter->regValue = (LwU32*)malloc(sizeof(LwU32));

        *(iter->regValue) = value;

        loggerNode = createAndAppendLoggerNode(iter, value);
        }
    }
}

/*!
 * @brief Unit test Infra gpu read function
 *
 * @param[in]      addr        address of register
 *
 * @return         value read
 *
 */
static LwU32
unitGpuReadReg
(
    LwU32 addr
 )
{

    readRegOp *readIter;
    observer *obsIter;
    LwBool wasWritten;
    LwU32 returlwalue;

    //check if readRegop list is empty
    if (readRegopHead == NULL)
    {
        returlwalue = regopReadRegister(addr, &wasWritten);

        //check if this register was previously written
        if (wasWritten)
        {
            return returlwalue;
        }

        //not previously written, error
        else
        {
            UNIT_ASSERT(FALSE);
            return 0xACDCACDC;
        }
    }

    //search readRegop node for this reg, if present
    else
    {
        readIter = findRegopNode(readRegopHead, addr, READ);

        //node present
        if (readIter)
        {
            //is there a callback
            if (readIter->callback)
            {
                return readIter->callback(readIter->callbackArg);
            }

            //is there will return kind of functionality
            if (readIter->observerHead)
            {
                //
                //check if ther is an always return kind
                //of finctionality, if present return the value
                //
                if ((readIter->observerHead->count) == 0)
                {
                   return (readIter->observerHead->value);
                }

                //reurn on the basis of count
                else
                {
                    readIter->timesCalled += 1;

                    returlwalue = (readIter->observerHead->value);

                    if ( (readIter->timesCalled) == (readIter->observerHead->count) )
                    {
                        obsIter = (readIter->observerHead);
                        (readIter->observerHead) = ((readIter->observerHead->next));
                        free(obsIter);
                        readIter->timesCalled = 0;
                    }

                    return returlwalue;
                }
            }
        }
    }

    //
    //there is no value expected to return, check if
    //we can find ssomethinf via written value
    //
    returlwalue = regopReadRegister(addr, &wasWritten);

    //check if this register was previously written
    if (wasWritten)
    {
        return returlwalue;
    }

    //not previously written, error
    else
    {
        UNIT_ASSERT(FALSE);
        return 0xACDCACDC;
    }
}

/*!
 * @brief wrapper around unitGpuReadReg, for 32 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @return         value read
 *
 */
LwU32
unitGpuReadRegister032
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr
 )
{
    return unitGpuReadReg(addr);
}

/*!
 * @brief wrapper around unitGpuWriteReg, for 32 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @param[in]      value  value to be written
 *
 */
void
unitGpuWriteRegister032
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr,
    LwU32           value
 )
{
    unitGpuWriteReg(addr, value);
}

/*!
 * @brief wrapper around unitGpuReadReg, for 16 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @return         value read
 *
 */
LwU16
unitGpuReadRegister016
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr
 )
{
    return (LwU16)unitGpuReadReg(addr);
}

/*!
 * @brief wrapper around unitGpuWriteReg, for 16 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @param[in]      value  value to be written
 *
 */
void
unitGpuWriteRegister016
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr,
    LwU16           value
 )
{
    unitGpuWriteReg(addr, value);
}

/*!
 * @brief wrapper around unitGpuReadReg, for 8 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @return         value read
 *
 */
LwU8
unitGpuReadRegister008
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr
 )
{
    return (LwU8)unitGpuReadReg(addr);
}

/*!
 * @brief wrapper around unitGpuWriteReg, for 8 bit reg
 *
 * @param[in]      pGpu   pointer to gpu obj, dont care for unit test
 *
 * @param[in]      deviceIndex   Identifies the device to be accessed
 *
 * @param[in]      addr   address of register
 *
 * @param[in]      value  value to be written
 *
 */
void
unitGpuWriteRegister008
(
    OBJGPU         *pGpu,
    DEVICE_INDEX    deviceIndex,
    LwU32           addr,
    LwU8            value
 )
{
    unitGpuWriteReg(addr, value);
}

/*!
 * @brief wrapper around unitGpuReadReg, for pmu-sw
 *
 * @param[in]      addr   address of register
 *
 * @return         value read
 *
 */
LwU32
unitPmuReadRegister
(
     LwUPtr addr
 )
{
    return unitGpuReadReg((LwU32)addr);
}

/*!
 * @brief wrapper around unitGpuWriteReg, for pmu-sw
 *
 * @param[in]      addr   address of register
 *
 * @param[in]      value  value to be written
 *
 */
void
unitPmuWriteRegister
(
     LwUPtr addr,
     LwUPtr value
 )
{
    unitGpuWriteReg((LwU32)addr, (LwU32)value);
}

