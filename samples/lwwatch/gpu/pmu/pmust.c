/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmust.c
 * @brief WinDbg Extension for PMU, printing out stack trace of PMU
 *
 *
 * This file contains implementation of LwWatch PMUST command which is used to dump stack trace
 * of a HALTED PMU (all FALCONs will be supported once we generalize the code).
 *
 * It provides three interfaces for other functions to call:
 *
 * pmustLoad             ---  Load and parse symbol files that are required to stack trace the stack,
 *                            build a cache to improve the speed of stack parsing
 * pmustUnload           ---  Unload the cache which is built by pmustLoad
 * pmustPrintStacktrace  ---  Print out the stacktrace provided the stack content, stack size and ucodeVersion
 *                            if pmustLoad is not called before, this command will try to call pmustLoad to do
 *                            implicit intialization (error msg will be printed out if this step fails)
 *
 * NOTE:
 *
 * 1. Lwrrently pmust understands five stack operations:
 *
 *    pushm, popm, push, pop, addsp 
 *    from the defintion in Falcon Architecture Specification (24-Jul-11).
 *
 *    Any updates to current stack operations or if new stack operations are added,
 *    pmust MUST be updated in order to operate correctly.
 *
 * 2. pmust relies on the following symbols defined in PMU/RTOS, any modifications
 *    to the symbols, including their names or meanings, pmust will FAIL:
 *
 *    _isr_stack_start, _isr_stack_end    defined in uproc/libs/rtos/src/osdebug.c
 *                                               and uproc/build/templates/gt_sections.ld
 *     pxLwrrentTCB                       defined in tools/resman/pmu/pmusim/fixture/pmusw/SimOpenRTOS/Source/
 *
 * 3. pmust cannot cope with code that has stack operations inside branches
 *    Eg.  
 *         if (condition)
 *         {
 *              push / pop / addsp / pushm /popm
 *         }
 *         ....
 *         
 *    Since for now, it is impossible for pmust to know if this condition is true or false,
 *    pmust may parse the stack incorrectly in this case.(lwrrently condition is assumed to be true)
 *    
 * 
 * Implementation Details:
 *
 * pmust uses information from .objdump and .nm files to parse the content
 * of the stack and print out the stack trace.
 *
 * The situations pmust can cope with can be divided into 5 categories:
 *
 * 1. Print out stacktrace for call instructions through immediate value.
 *    eg. call 0x1234
 * 2. Recongize calls that are done through reigsters.
 *    eg. call a10
 * 3. Recongize if the current halt happens in ISR. In this case, pmust should
 *    print out the stack trace for both ISR and the stacktrace of the task 
 *    before trapping into ISR.
 * 4. Print out the stack trace correctly in the case we have IMEM error.
 * 5. When the task control block is corrupted, namely when we don't know
 *    where the stack bottom is. In this case, parse the stack as much as we 
 *    can, stop when an error is encountered.
 *     
 * Process of the parsing procedure:
 *
 * 1. Parse the objdump file to build such structure for each function
 *    
 *    {  
 *      function_name,
 *      entryPoint,
 *      all stack operations in the function, including the address of the instr
 *      all call instructions in the function and their targets
 *    }
 *
 * 2. Given the PC, SP and the whole stack, we can start processing the stack
 *    based on the information in 1.
 * 
 *    (1) Detect if the halt is inside ISR by comparing SO with _isr_stack_start and
 *        _isr_stack_end. If yes, print out some msg, mark a note here so that
 *        pmust will also remember to parse the task which trapped into ISR.
 *    (2) Start from the value of the pc register
 *        current address = pc
 *        current stack pointer = sp
 *
 *        loop the following operations until the bottom of the stack 
 *        is reached or error is found.
 *
 *        . If current stack pointer has reached the bottom of the stack, stop!
 *        . Found the function structure which the current address belongs to
 *        . Loop over all the stack operations in the function between 
 *          entrypoint and the current address.
 *        . Callwate how many stack space is used -- totalStackUsed
 *          set current stack pointer += totalStackUsed
 *        . Fetch the return address from the current stack pointer,
 *        . Validate the correctness of the address fetched, by go to the
 *          corresponding function structure, find if the current function
 *          can be called from that function.
 *
 *          Should be noticed that, if a function ever call through register,
 *          pmust considers it can call any functions in the whole file.
 *        . Stop if not valid, otherwise set current address = new return addr 
 *          fetched, continue the loop.
 *
 * 3. For cases the PC is at ISR, it means that we need to find the original
 *    task before entering into ISR.
 *
 *    The current implementation, RTOS store the original stack pointer at the
 *    address that pxLwrrentTCB points to. So when stacktrace finished at ISR, fetch the 
 *    stack pointer from pxLwrrentTCB, continue the process.
 *
 * Assumptions on format of objdump file:
 *
 * 1. Function name line format
 *    (1) The line should only contains entry address, function name and characters '<', '>' and ':'
 *    (2) format:
          EntryAddr <FuncName>:\n
 *    (3) Between entry addrss and function name, there can be numerous number of spaces or tabs or both.
 *        EntryAddr should be in hex format, it is not required to have the starting 0s.
 *
 *      eg.
 *      00000676 <_prvIdleTask>:
 *
 *
 * 2. Instruction line format
 *    (1) Format:
 *        Addr:   instr_in_byte_format    Instr;\n
 *    (2) Addr must be in hex format, the trailing ";" cannot be omitted.
 *    (3) Between the three items, there can be numerous number of spaces or tabs or both.
 *
 *        eg.
 *        676:    f9 00                     push a0;
 *
 * 3. Whole function format
 *    (1) The first line should be the function name line.
 *    (2) All the instruction lines should follow the function name line,
 *        in asecending order of their addresses.
 *
 * 4. Any changes to the format will break the command pmust. So pmust must
 *    be updated when the objdump format is updated to be different from the
 *    above description.
 */

/* ------------------------ Includes --------------------------------------- */
#include "pmu.h"
#include "lwsym.h"

/* ------------------------ Types definitions ------------------------------ */
typedef struct PmuFunc PmuFunc;
typedef struct PmuFuncCall PmuFuncCall;
typedef struct LinkedListElem LinkedListElem;

/** @struct LinkedlistElem
 *
 *  structure to store items in a linked list
 */
struct LinkedListElem
{
    void            *pObj;
    LwU32            extraField;
    LinkedListElem  *pNext;
};

/** @enum CALL_TYPE
 *
 *  Used to represent if a function call is actually happened at that address
 */
typedef enum
{
    CALL_TYPE_ILWALID      = 0,
    CALL_TYPE_VALID        = 1,
    CALL_TYPE_CALLFROMREG  = 2
} CALL_TYPE;

/** @enum STACK_OP 
 *
 *  Three types of stack operations lwrrently cared about
 *  Note: "wspr SP value" is the instruction that change sp, but could not be tracked now.
 */
typedef enum
{
    STACK_OP_PUSH,
    STACK_OP_POP,
    STACK_OP_PUSHM,
    STACK_OP_POPM,
    STACK_OP_ADDSP
} STACK_OP;

/** @struct StackOp
 *
 *  Structure to represent things about a stack operation
 */
typedef struct
{
    STACK_OP      opType;
    LwU32         addr;
    short         value;
    LwBool        bRet;
} StackOp;

/** @struct PmuCallFunc
 *
 *  All things about a function call
 */
struct PmuFuncCall
{
    // Target fucntion we call
    PmuFunc  *pFunc;

    // Start Address of the call instruction
    LwU32     callAddr;

    // Start address of instruction followed by this call
    LwU32     pNextAddr;
};

/** @struct PmuFunc
 *
 *  Abstraction of a function in objdump
 */
struct PmuFunc
{
    char                funcName[64];
    char                funlwpName[64];
    LwU32               entryPoint;
    LwU32               lastAddr;
    PmuFuncCall        *pOutFuncs;
    LwU16               numOutFuncs;
    StackOp            *pStackOps;
    LwU32               numStackOps;
    PmuFunc            *pNext;
};

/** @struct StackOpMap
 *
 *  Structure used to map stack operation string to stack type
 */
typedef struct
{
    const char *upperKeyword;
    STACK_OP    opType;
} StackOpMap;

/* ------------------------ Static variables ------------------------------- */

//
// Variable to store the function table, including 
//  1. information of all functions
//  2. information of all function calls inside each function
//  3. information of all stack operation inside function 
//
static PmuFunc **ppPmuFuncTable = NULL;

/// Total number of functions we have
static LwU32 NumPmuFuncs = 0;

// Do we have loaded the objdump file and successfully parsed it?
static BOOL bPmuFuncLoaded = FALSE;

// The sequence here is important pushm and popm should be check before push and pop
static StackOpMap stackOpMap[] = 
{
    {"PUSHM",  STACK_OP_PUSHM},
    {"POPM",   STACK_OP_POPM},
    {"PUSH",   STACK_OP_PUSH},
    {"POP",    STACK_OP_POP},
    {"ADDSP",  STACK_OP_ADDSP}
};

// The string used to recognize a function call instruction
static char *callInstUp = " CALL ";

// Length of the string in variable callInstUp
static LwU32 callInstUpLen = 6;

// How many stack operations can be pasrsed
static int numStackOps = (int)(sizeof(stackOpMap) / sizeof(StackOpMap));

/* ------------------------ Function Prototypes ---------------------------- */
static PmuFunc      *pmuCreateFunc                  (const char *pLine);
static BOOL          pmuFuncListToArray             (PmuFunc ***pppPmuFuncTable, PmuFunc *pPmuFuncList, LwU32 numPmuFuncs);
static int           addrCmp                        (LwU32 targetAddr, PmuFunc *pPmuFunc, BOOL bExact);
static BOOL          pmuFuncListToArray             (PmuFunc ***pppPmuFuncTable, PmuFunc *pPmuFuncList, LwU32 numPmuFuncs);
static PmuFunc      *pmuFuncSearch                  (PmuFunc **ppPmuFuncTable, LwU32 numPmuFuncs, LwU32 addr, BOOL bExact);
static StackOp      *parseStackOp                   (char *pLine, STACK_OP opType);
static PmuFuncCall  *parseCallFunc                  (char *pLine, PmuFunc **ppPmuFuncTable, LwU32 numPmuFuncs);
static BOOL          pmuFuncParseCode               (PmuFunc  **ppPmuFuncTable, LwU32 numPmuFuncs, LwU32 indexFunc, char *pFuncLines, char *pFinalChar);
static BOOL          objdumpFirstPass               (const LwU8 *pFileBuffer, LwU32 fileBufferSize, LwU32 ucodeVersion, BOOL bVerbose);
static BOOL          objdumpSecondPass              (LwU8 *pFileBuffer, LwU32  fileBufferSize, BOOL bVerbose);
static void          printStacktraceHelper          (LwU8 *stackContent, LwU32 contentSize, LwU32 stackSize, LwS32 stackIdx, LwU32 lwrPC, PmuFunc **ppPmuFuncTable, LwU32 numPmuFuncs, LwU32 frameIdx, LwU32 numNotCalls);


/* ------------------------ Defines ---------------------------------------- */
#define ISDIGIT(x) ((x) >= '0' && (x) <= '9')
#define ISALPHA(x) (((x) >= 'a' && (x) <= 'z') || ((x) >= 'A' && (x) <= 'Z'))


/*!
 * @brief  Parse the function declaration line, create a corresponding PmuFunc structure
 *
 * @param[in]  pLine  A line in the objdump file which is the first line(declaration) of a function
 *
 * @return     Dynamically alloated PmuFunc structure containing ONLY the function name and entrypoint
 */
static PmuFunc *
pmuCreateFunc
(
    const char *pLine
)
{
    LwU32     index = 0;
    LwU32     idxStart, strSize;
    long      addr;
    PmuFunc  *pNewFunc = (PmuFunc *)malloc(sizeof(PmuFunc));

    memset((void *)pNewFunc, 0, sizeof(PmuFunc));

    while (pLine[index] != ' ')
        ++index;

    addr = strtol(pLine, NULL, 16);

    while (pLine[index] != '<')
        ++index;

    ++index;
    idxStart = index;

    while (pLine[index] != '>')
        ++index;

    strSize = index - idxStart;

    pNewFunc->entryPoint = (LwU32) addr;
    memcpy(pNewFunc->funcName, &pLine[idxStart], strSize);
    pNewFunc->funcName[strSize] = '\0';

    for (index = 0; index < strSize; ++index)
        pNewFunc->funlwpName[index] = (char)toupper(pNewFunc->funcName[index]);
    pNewFunc->funlwpName[strSize] = '\0';

    return pNewFunc;
}

/*!
 * @brief  Colwert linked list to array. The reason for this is that binary search
 *         can be used for searching a function by address.
 *
 * @param[out] pppPmuFuncTable  The colwerted array
 * @param[in]  pPmuFuncList    Linkedlist storing all function information
 * @param[in]  numPmuFuncs    How many functions are in the list
 *
 * @return     TRUE
 *      The function addresses are in a correct order,
 * @return     FALSE
 *      The functions are not in a correct order, meaning objdump may have changed
 */
static BOOL
pmuFuncListToArray
(
    PmuFunc ***pppPmuFuncTable,
    PmuFunc   *pPmuFuncList,
    LwU32      numPmuFuncs
)
{
    PmuFunc  **pFuncTable = (PmuFunc **)malloc(sizeof(PmuFunc *) * numPmuFuncs);
    int        index;
    LwU32      prevFuncAddr = pPmuFuncList->entryPoint + 1;
    BOOL       bValid  = TRUE;

    for (index = numPmuFuncs - 1; index > -1; --index)
    {
        if (pPmuFuncList->entryPoint >= prevFuncAddr)
            bValid = FALSE;

        prevFuncAddr = pPmuFuncList->entryPoint;
        pFuncTable[index] = pPmuFuncList;
        pPmuFuncList = pPmuFuncList->pNext;
    }

    *pppPmuFuncTable = pFuncTable;

    return bValid;
}

/*!
 * @brief  Helper function for binary search.
 *         Help binary search to locate which function the address belongs to.
 *
 * @param[in]  targetAddr   The address being interested in.
 * @param[in]  pPmuFunc     The function being compared
 * @param[in]  bExact       For bExact is TRUE, compare with the entrypoint of the function
 *                          For bExact is FALSE, considered the function as a set of addresses
 *                          [entryPoint, entryPoint of pNextFunction)
 * @return     -1
        if targetAddr < pmuFunc
   @return      0
        if targetAddr == pmuFunc
   @return      1
        if targetAddr > pmuFunc
   @return     -2
        if error happen
 */
static LwS32
addrCmp
(
    LwU32     targetAddr,
    PmuFunc  *pPmuFunc,
    BOOL      bExact
)
{
    if (bExact)
    {
        if (targetAddr == pPmuFunc->entryPoint)
            return 0;
        else if (targetAddr > pPmuFunc->entryPoint)
            return 1;
        else
            return -1;
    }
    else
    {
        if ((targetAddr >= pPmuFunc->entryPoint) && (targetAddr < pPmuFunc->lastAddr))
            return 0;
        else if (targetAddr < pPmuFunc->entryPoint)
            return -1;
        else if (targetAddr > pPmuFunc->lastAddr)
            return 1;
        else
            return -2;
    }
}

/*!
 * @brief  Binary search for a function based on the address
 * 
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 * @param[in]  addr            Address we are interested in, that which function it belongs to
 * @param[in]  bExact          If we are interested in the entryPoint or the whole function
 *
 * @return    Pointer to the function being found
 *       The target function is found
 * @return    NULL 
 *       No match is found.
 */
static PmuFunc *
pmuFuncSearch
(
    PmuFunc  **ppPmuFuncTable,
    LwU32      numPmuFuncs,
    LwU32      addr,
    BOOL       bExact
)
{
    LwU32 start = 0;
    LwU32 end   = numPmuFuncs - 1;

    for (; ;)
    {
        LwU32 mid = (end - start) / 2 + start;
        LwS32 cmpRes;

        if (start > end)
        {
            return NULL;
        }
        else if (start == end)
        {
            if (addrCmp(addr, ppPmuFuncTable[start], bExact) == 0)
                return ppPmuFuncTable[start];
            else 
                return NULL;
        }

        cmpRes = addrCmp(addr, ppPmuFuncTable[mid], bExact);
        if (cmpRes == 0)
            return ppPmuFuncTable[mid];
        else if (cmpRes == -1)
            end = mid - 1;
        else if (cmpRes == 1)
            start = mid + 1;
        else
            return NULL;
    }
    return NULL;
}

/*!
 * @brief  Parse the stack instruction string, create a corresponding StackOp structure
 *         to represent this stack operation.
 *
 * @param[in]  pLine    A line in the objdump file which is the decode of a stack operation
 * @param[in]  opType   Type of the stack operation
 *
 * @return     Dynamically alloated StackOp structure containing all the information
 *             we want from the stack operation
 */
static StackOp *
parseStackOp
(
    char         *pLine,
    STACK_OP      opType
)
{
    char    *pNextChar;
    LwU32    addr = (LwU32)strtol(pLine, &pNextChar, 16);
    StackOp *pStackOp;
    LwU32    retVal;

    if (pNextChar[0] != ':')
    {
        dprintf("Unexpected error in function %s\n", __FUNCTION__);
        return NULL;
    }

    pStackOp = (StackOp *)malloc(sizeof(StackOp));
    pStackOp->addr = addr;
    pStackOp->opType = opType;
    pStackOp->value = 0;
    pStackOp->bRet = LW_FALSE;

    if (opType == STACK_OP_ADDSP)
    {
        LwU32 index = (LwU32)strlen(pLine);

        --index;
        while ((pLine[index] != ' ') && (index != 0))
            --index;
        pStackOp->value = (short)strtol(&pLine[index], &pNextChar, 16);
    }
    else if (opType == STACK_OP_PUSHM)
    {
        LwU32 index = (LwU32)strlen(pLine);

        --index;
        while ((pLine[index] != 'A') && (index != 0))
            --index;
        pStackOp->value = ((short)strtol(&pLine[index + 1], &pNextChar, 10) + 1) * 4;
    }
    else if (opType == STACK_OP_POPM)
    {
        LwU32 index = (LwU32)strlen(pLine);
        LwU32 numSpaces = 0;

        --index;
        while ((pLine[index] != 'A') && (index != 0))
        {
            if (pLine[index] == ' ')
                ++numSpaces;
            --index;
        }

        pStackOp->value = ((short)strtol(&pLine[index + 1], &pNextChar, 10) + 1) * 4;

        if (numSpaces == 2)
        {
            char *pTemp = pNextChar;

            retVal = strtol(pTemp, &pNextChar, 16);
            if (retVal == 1)
            {
                // popm x, 1, x meaning return to the caller after the pop
                pStackOp->bRet = LW_TRUE;
            }

            pNextChar = pTemp;
        }
        pStackOp->value += (short)strtol(pNextChar, NULL, 16);
    }

    return pStackOp;
}

/*!
 * @brief  Parse the call instruction string, create a corresponding CALL structure to 
 *         store all informations about this call.
 *
 * @param[in]  pLine           A line in the objdump file which is the decode of a call instruction
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 *
 * @return     Dynamically alloated PmuFunc structure containing all the information
 *             we want from the call instruction
 */
static PmuFuncCall *
parseCallFunc
(
    char      *pLine,
    PmuFunc  **ppPmuFuncTable,
    LwU32      numPmuFuncs
)
{
    char         *pNextChar;
    char         *pSubStr;
    LwU32         callAddr = (LwU32)strtol(pLine, &pNextChar, 16);
    LwU32         targetAddr;
    PmuFuncCall  *pCallFunc;
    PmuFunc      *pTargetFunc;
    int           index;
    int           tokens;
    int           idx;

    if (pNextChar[0] != ':')
    {
        dprintf("Unexpected error in function %s\n", __FUNCTION__);
        return NULL;
    }

    pSubStr = strstr(pLine, callInstUp);
    idx = callInstUpLen;
    while (pSubStr[idx] == ' ')
        ++idx;

     /* Eg. call a10   The address is in register a10, so that pTargetFunc is unknown, set to NULL then*/
     if (pSubStr[idx] != '0')
        pTargetFunc = NULL;
     else
     {
        targetAddr = (LwU32)strtol(&pSubStr[callInstUpLen], NULL, 16);
        pTargetFunc = pmuFuncSearch(ppPmuFuncTable, numPmuFuncs, targetAddr, TRUE);
    }


    pCallFunc = (PmuFuncCall *)malloc(sizeof(PmuFuncCall));
    pCallFunc->callAddr = callAddr;
    pCallFunc->pFunc = pTargetFunc;

    index = 0;
    while (pNextChar[index] == ' ')
        ++index;

    tokens = 0;
    do
    {
        if (pNextChar[index] == ' ')
            ++tokens;
        ++index;
    } while (!((pNextChar[index] == ' ') && (pNextChar[index - 1] == ' ')));

    pCallFunc->pNextAddr = pCallFunc->callAddr + tokens;

    return pCallFunc;
}

/*!
 * @brief  The function to parse the whole decode of a function and filled all the fields
 *         in the corresponding PmuFunc structure
 *
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 * @param[in]  indexFunc       Index of the function being parsed in ppPmuFuncTable
 * @param[in]  pFuncLines      All the text of the function.
 * @param[in]  pFinalChar      Pointer to the last char of the text
 *
 * @return     TRUE
 *      Parsing succeeded
 * @return     FALSE
 *      Parsing failed
 */
static BOOL
pmuFuncParseCode
(
    PmuFunc  **ppPmuFuncTable,
    LwU32      numPmuFuncs,
    LwU32      indexFunc,
    char      *pFuncLines,
    char      *pFinalChar
)
{
    int              cursor = 0;
    int              index = 0;
    char             line[256];
    BOOL             bBlank = TRUE;
    PmuFunc         *pLwrFunc = NULL;
    LinkedListElem  *pStackOps = NULL;
    LinkedListElem  *pOutFuncs = NULL;
    LwU32            lastAddr = 0;
    BOOL             bLastFunc;

    if (indexFunc == numPmuFuncs - 1)
        bLastFunc = TRUE;
    else
        bLastFunc = FALSE;

    while (&pFuncLines[cursor] != pFinalChar)
    {
        bBlank = TRUE;
        while ((&pFuncLines[cursor] != pFinalChar) && (pFuncLines[cursor] != '\n') && (pFuncLines[cursor] != '\0'))
        {
            if ((bBlank) && (ISDIGIT(pFuncLines[cursor]) || ISALPHA(pFuncLines[cursor])))
                bBlank = FALSE;
            line[index++] = pFuncLines[cursor++];
        }

        if (pFuncLines[cursor] == '\0')
            break;

        if (!bBlank)
        {
            line[index] = '\0';

            // last characters of line >:\0  
            // then it will be the line for function name and start addr
            //
            if ((line[index - 1] == ':') && (line[index - 2] == '>'))
            {
                if (pLwrFunc == NULL)
                {
                    LwU32 addr = (LwU32)strtol(line, NULL, 16);
                    if (addr == ppPmuFuncTable[indexFunc]->entryPoint)
                    {
                        pLwrFunc = ppPmuFuncTable[indexFunc];

                        if (!bLastFunc)
                        {
                            pLwrFunc->lastAddr = ppPmuFuncTable[indexFunc + 1]->entryPoint - 1;
                        }
                    }
                    else
                    {
                        dprintf("Unexpected error in function %s\n", __FUNCTION__);
                        return FALSE;
                    }
                }
                else
                {
                    dprintf("Unexpected error in function %s, possibly output format of objdump is changed and LwWatch is not updated accordingly.\n", __FUNCTION__);
                    return FALSE;
                }
            }
            // last character of line ; , and begins with two spaces, it is decoded line
            else if ((line[index - 1] == ';') && (line[0] == ' ') && (line[1] == ' '))
            {
                int iter;

                // We need to keep track of the last addrres
                if (bLastFunc)
                {
                    char *pNextChar;
                    int   lidx;
                    LwU32 tokens;

                    lastAddr = (LwU32)strtol(line, &pNextChar, 16);
                    lidx = 0;
                    while (pNextChar[lidx] == ' ')
                        ++lidx;

                    tokens = 0;

                    do
                    {
                        if (pNextChar[lidx] == ' ')
                            ++tokens;
                        ++lidx;
                    } while (!((pNextChar[lidx] == ' ') && (pNextChar[lidx - 1] == ' ')));

                    lastAddr += tokens;
                }

                for (iter = 0; line[iter] != '\0'; ++iter)
                    line[iter] = (char)toupper(line[iter]);

                for (iter = 0; iter < numStackOps; ++iter)
                    if (strstr(line, stackOpMap[iter].upperKeyword) != NULL)
                    {
                        LinkedListElem *pNewElem    = (LinkedListElem *)malloc(sizeof(LinkedListElem));
                        StackOp        *newStackOp = parseStackOp(line, stackOpMap[iter].opType);

                        pNewElem->pObj = (void *)newStackOp;
                        pNewElem->pNext = pStackOps;
                        pStackOps = pNewElem;
                        ++pLwrFunc->numStackOps;
                        break;
                    }

                if (strstr(line, callInstUp) != NULL)
                {
                    PmuFuncCall      *pCallFunc;
                    LinkedListElem   *pNewElem = (LinkedListElem *)malloc(sizeof(LinkedListElem));

                    ++pLwrFunc->numOutFuncs;

                    pCallFunc = parseCallFunc(line, ppPmuFuncTable, numPmuFuncs);
                    pNewElem->pObj = (void *)pCallFunc;
                    pNewElem->pNext = pOutFuncs;
                    pOutFuncs = pNewElem;
                } // ends if ((pSubStr = strstr(line, callInstUp)) != NULL)
            } // ends pCallFunc != NULL
        } // ends !bBlank
        index = 0;
        ++cursor;
    } // ends while loop

    if (pLwrFunc == NULL)
    {
        dprintf("Unexpected error in function %s, possibly output format of objdump is changed and LwWatch is not updated accordingly.\n", __FUNCTION__);
        return FALSE;
    }

    if (bLastFunc)
        pLwrFunc->lastAddr = lastAddr;

    if (pLwrFunc->numOutFuncs > 0)
    {
        pLwrFunc->pOutFuncs = (PmuFuncCall *)malloc(sizeof(PmuFuncCall) * pLwrFunc->numOutFuncs);
        for (index = pLwrFunc->numOutFuncs - 1; index > -1; --index)
        {
            LinkedListElem* pTemp;
            pLwrFunc->pOutFuncs[index] = *(PmuFuncCall *)pOutFuncs->pObj;
            pTemp = pOutFuncs->pNext;
            free(pOutFuncs);
            pOutFuncs = pTemp;
        }
    }

    if (pLwrFunc->numStackOps > 0)
    {
        pLwrFunc->pStackOps = (StackOp *)malloc(sizeof(StackOp) * pLwrFunc->numStackOps);
        for (index = pLwrFunc->numStackOps - 1; index > -1; --index)
        {
            LinkedListElem* pTemp;

            pLwrFunc->pStackOps[index] = *(StackOp *)pStackOps->pObj;
            pTemp = pStackOps->pNext;
            free(pStackOps);
            pStackOps = pTemp;
        }
    }

    return TRUE;
}

/*!
 * @brief  First pass only creates a struct with the function name, start address, but not other infos
 *         The function expects the function is the order from low address to high address.
 *
 * @param[in]  pFileBuffer     Buffer storing the whole text of objdump
 * @param[in]  fileBufferSize  Buffer size
 * @param[in]  uCodeVersion    ucode version
 * @param[in]  bVerbose        Print out all the intermediate parsing info or not
 *
 * @return     TRUE
 *      Parsing succeeded
 * @return     FALSE
 *       Parsed failed
 */
static BOOL 
objdumpFirstPass
(
    const LwU8 *pFileBuffer,
    LwU32       fileBufferSize,
    LwU32       ucodeVersion,
    BOOL        bVerbose
)
{
    LwU32        index  = 0;
    char         line[256];
    LwU32        fileBufferLwrsor  = 0;
    const char  *verStr = "AppVersion: ";
    const size_t verLen = strlen(verStr);
    PmuFunc     *pPmuFuncList = NULL;

    // Now read the file in character by character.  Keep track of newlines
    // and whitespace (to count tokens). Each line will represent a symbol.
    // The token count is needed since not all symbols will be listed with
    // their size and the token count is the easiest way to detect those cases.
    //
    while (fileBufferLwrsor < fileBufferSize)
    {
        line[index] = pFileBuffer[fileBufferLwrsor++];

        if (line[index] != '\n')
        {
            index++;
        }
        else
        {
            line[index+1] = '\0';
            //
            // If the line starts with "AppVersion: ", compare it to the
            // requested version.  We need to specify the length exactly so
            // that it still matches although "line" doesn't have a NUL
            // terminator there.
            //
            if (!strncmp(line, verStr, verLen))
            {
                LwU32 version = 0;
                int ret = sscanf(line + verLen, "%u", &version);
                if ((ret) && (ret != EOF))
                {
                    if (version != ucodeVersion)
                    {
                        dprintf("lw: Warning: PMU ucode version mismatch.\n");
                        dprintf("lw:          on chip: %d, on disk: %d\n",
                                ucodeVersion, version);
                    }
                }
                else
                {
                    dprintf("lw: Error: Unable to determine ucode version on disk.\n");
                }
            }
            else if ((index > 1) && (line[index - 1] == ':') && (line[index - 2] == '>'))
            {
                PmuFunc *pNewFunc = pmuCreateFunc(line);

                pNewFunc->pNext = pPmuFuncList;
                pPmuFuncList = pNewFunc;
                if (bVerbose)
                    dprintf("%d 0x%x : %s\n", NumPmuFuncs, pNewFunc->entryPoint, pNewFunc->funcName);
                ++NumPmuFuncs;
            }
            index  = 0;
        }
    }

    return pmuFuncListToArray(&ppPmuFuncTable, pPmuFuncList, NumPmuFuncs);
}

/*!
 * @brief  Second pass creates the call relationships and track the stack operations 
 *
 * @param[in]  pFileBuffer     Buffer storing the whole text of objdump
 * @param[in]  fileBufferSize  Buffer size
 * @param[in]  bVerbose        Print out all the intermediate parsing info or not
 *
 * @return     TRUE
 *      Parsing succeeded
 *             FALSE
 *      Parsing failed
 */
static BOOL 
objdumpSecondPass
(
    LwU8  *pFileBuffer,
    LwU32  fileBufferSize,
    BOOL   bVerbose
)
{
    LwU32    index  = 0;
    LwU32    indexFunc = 0;
    char     line[256];
    LwU32    fileBufferLwrsor  = 0;
    char    *pFuncStart;
    char    *pFuncEnd;
    BOOL     parseFuncRes = FALSE;

    pFuncStart = NULL;

    while (fileBufferLwrsor < fileBufferSize)
    {
        line[index] = pFileBuffer[fileBufferLwrsor++];

        if (line[index] == '\0')
        {
            break;
        }
        else if (line[index] != '\n')
        {
            index++;
        }
        else
        {
            line[index+1] = '\0';

            if ((index > 1) && (line[index - 1] == ':') && (line[index - 2] == '>'))
            {
                if (pFuncStart != NULL)
                {
                    pFuncEnd = (char *)&pFileBuffer[fileBufferLwrsor - index - 1];
                    parseFuncRes = pmuFuncParseCode(ppPmuFuncTable, NumPmuFuncs, indexFunc, pFuncStart, pFuncEnd);

                    if (!parseFuncRes)
                        break;

                    // Debug output
                    if (bVerbose)
                    {
                        PmuFunc *pFunc = ppPmuFuncTable[indexFunc];
                        LwU32    index;

                        dprintf("lw: 0x%x %s\n", pFunc->entryPoint, pFunc->funlwpName);
                        for (index = 0; index < pFunc->numOutFuncs; ++index)
                        {
                            dprintf("0x%x call : ", pFunc->pOutFuncs[index].callAddr);
                            if (pFunc->pOutFuncs[index].pFunc != NULL)
                                dprintf("0x%x\n", pFunc->pOutFuncs[index].pFunc->entryPoint);
                            else
                                dprintf(" register\n");
                        }
                        for (index = 0; index < pFunc->numStackOps; ++index)
                            dprintf("0x%x stack  %s\n", pFunc->pStackOps[index].addr, stackOpMap[pFunc->pStackOps[index].opType].upperKeyword);

                    }
                    ++indexFunc;
                }
                pFuncStart = (char *)&pFileBuffer[fileBufferLwrsor - index - 1];
            }
            index  = 0;
        }
    }

    // Parse the last function
    if (parseFuncRes)
        parseFuncRes = pmuFuncParseCode(ppPmuFuncTable, NumPmuFuncs, indexFunc, pFuncStart, (char *)&pFileBuffer[fileBufferLwrsor - 1]);

    if (!parseFuncRes)
    {
        LwU32 index;

        dprintf("Unexpected error in function %s due to failure in parsing objdump\
                 text, please verify the code, it may need to update to format changed.\n", __FUNCTION__);

        // Clean all the things
        for (index = 0; index < indexFunc; ++index)
        {
            free(ppPmuFuncTable[index]->pStackOps);
            free(ppPmuFuncTable[index]->pOutFuncs);
            free(ppPmuFuncTable[index]);
        }

        bPmuFuncLoaded = FALSE;
        ppPmuFuncTable = NULL;
        NumPmuFuncs = 0;
    }

    // debug output
    if (bVerbose)
    {
        PmuFunc *pFunc = ppPmuFuncTable[indexFunc];
        LwU32    index;

        dprintf("lw: 0x%x %s\n", pFunc->entryPoint, pFunc->funlwpName);
        for (index = 0; index < pFunc->numOutFuncs; ++index)
        {
            dprintf("0x%x call : ", pFunc->pOutFuncs[index].callAddr);
            if (pFunc->pOutFuncs[index].pFunc != NULL)
                dprintf("0x%x\n", pFunc->pOutFuncs[index].pFunc->entryPoint);
            else
                dprintf(" register\n");
            }
            for (index = 0; index < pFunc->numStackOps; ++index)
                dprintf("0x%x stack  %s\n", pFunc->pStackOps[index].addr, stackOpMap[pFunc->pStackOps[index].opType].upperKeyword);
    }

    return TRUE;
}

/*!
 * @brief  Load the .objdump file
 *
 * @param[in]  pFilename       Full path to the objdump file, load from lwsym if NULL
 * @param[in]  uCodeVersion    ucode version
 * @param[in]  bVerbose        Print out all the intermediate parsing info or not
 *
 * @return     TRUE
 *      Load objdump file and parsing succeeded
 *             FALSE
 *      Failed to load objdump file or parsing failed
 */
BOOL
pmustLoad
(
    const char  *pFilename,
    LwU32        ucodeVersion,
    BOOL         bVerbose
)
{
    LwU8        *pFileBuffer     = NULL;
    LwU32        fileBufferSize  = 0;
    LwU32        index           = 0;
    LWSYM_STATUS status;
    BOOL         bStatus;
    char        *pFileExtension;

    if (bPmuFuncLoaded)
    {
        if (bVerbose)
        {
            dprintf("lw: Error: Symbols are already loaded. Please unload "
                    "first using \"pmust -u\".\n");
        }
        return FALSE;
    }

    pFileExtension = ".objdump";

    // If a filename is not specified, load it from the LwSym package
    if ((pFilename == NULL) || (pFilename[0] == '\0'))
    {
        char       *pLwsymFilename = NULL;
        const char *pUcodeName     = pPmu[indexGpu].pmuUcodeName();
        pLwsymFilename = malloc(strlen(LWSYM_VIRUTAL_PATH) +
                                strlen("pmusym/")          +
                                strlen(pUcodeName)         +
                                strlen(pFileExtension)          +
                                (size_t)1);

        sprintf(pLwsymFilename, "%s%s%s%s",
                                LWSYM_VIRUTAL_PATH,
                                "pmusym/",
                                pUcodeName,
                                pFileExtension);

        if (bVerbose)
            dprintf("lw: Loading symbol file: %s\n", pLwsymFilename);
        status = lwsymFileLoad(pLwsymFilename, &pFileBuffer, &fileBufferSize);
        free(pLwsymFilename);
    }
    else
    {
        FILE *pFile = fopen(pFilename, "r");
        if (pFile == NULL)
            status = LWSYM_FILE_NOT_FOUND;
        else
        {
            LwU32 fileSize;

            fseek(pFile, 0, SEEK_END);
            fileSize = (LwU32)ftell(pFile);
            fseek(pFile, 0, SEEK_SET);
            if ((pFileBuffer = (LwU8 *)malloc(sizeof(LwU8) * fileSize)) != NULL)
            {
                fileBufferSize = (LwU32)fread(pFileBuffer, sizeof(LwU8), fileSize, pFile);
                dprintf("%d %d\n", fileBufferSize, fileSize);
                status = LWSYM_OK;
            }
            else
            {
               status = LWSYM_BUFFER_OVERFLOW;
            }
            
            fclose(pFile);
        }
        if (bVerbose)
            dprintf("lw: Loading symbol file: %s\n", pFilename);
    }

    if (status != LWSYM_OK)
    {
        dprintf("lw: Error reading file (%s)\n", lwsymGetStatusMessage(status));

        return FALSE;
    }

    // Replace all \r neigbouring \n to \n
    for (index = 0; index < fileBufferSize; ++index)
        if ((pFileBuffer[index] == '\r') || (pFileBuffer[index] == '\0'))
        {
            pFileBuffer[index] = '\n';
        }
    pFileBuffer[fileBufferSize -1] = '\0';

    bStatus = objdumpFirstPass(pFileBuffer, fileBufferSize, ucodeVersion, bVerbose);

    if (bStatus)
        bStatus = objdumpSecondPass(pFileBuffer, fileBufferSize, bVerbose);

    if (!bStatus)
    {
        dprintf("lw:  Unable to parse objdump file, this may due to updated format of the objdump file.\n");
        pmustUnload();
    }
    else
    {
        if (bVerbose)
        {
            dprintf("\n");
            dprintf("lw: PMU symbols loaded successfully.\n");
            dprintf("lw:\n");
        }
        bPmuFuncLoaded = TRUE;
    }

    if (pFilename == NULL)
        lwsymFileUnload(pFileBuffer);
    else
        free(pFileBuffer);

    return bStatus;
}

/*!
 * @brief  Unload the .objdump file
 *
 * @return     TRUE
 *      Unloaded successfully
 * @return     FALSE
 *      Unload failed
 */
BOOL
pmustUnload()
{
    LwU32 index;

    for (index = 0; index < NumPmuFuncs; ++index)
    {
        if (ppPmuFuncTable[index]->pOutFuncs != NULL)
            free(ppPmuFuncTable[index]->pOutFuncs);

        if (ppPmuFuncTable[index]->pStackOps != NULL)
            free(ppPmuFuncTable[index]->pStackOps);

        free(ppPmuFuncTable[index]);
    }

    bPmuFuncLoaded = FALSE;
    ppPmuFuncTable = NULL;
    NumPmuFuncs = 0;

    dprintf("\n");
    dprintf("lw: PMU objdump unloaded successfully.\n");
    dprintf("lw:\n");

    return TRUE;
}

/*!
 * @brief  Validate that if the function call is actually from the one found in stacktrace
 *
 * @param[in]  pCalledFunc     The function being called
 * @param[in]  pcAfterRet      The return address of the pCalledFunc
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 *
 * @return     TRUE
 *      The call relation is correct
 * @return     FALSE
 *      The call relation is not correct
 */
static CALL_TYPE 
validateCallRelation
(
    PmuFunc  *pCalledFunc,
    LwU32     pcAfterRet,
    PmuFunc **ppPmuFuncTable,
    LwU32     numPmuFuncs
)
{
    PmuFunc *originFunc = pmuFuncSearch(ppPmuFuncTable, numPmuFuncs, pcAfterRet, FALSE);
    LwU32    index;

    if (originFunc == NULL)
        return FALSE;

    for (index = 0; index < originFunc->numOutFuncs; ++index)
    {
        if (originFunc->pOutFuncs[index].pNextAddr == pcAfterRet)
        {
            if (originFunc->pOutFuncs[index].pFunc == pCalledFunc)
                return CALL_TYPE_VALID;
            else if (originFunc->pOutFuncs[index].pFunc == NULL)
                return CALL_TYPE_CALLFROMREG;
            else
                return CALL_TYPE_ILWALID;
        }
    }

    return CALL_TYPE_ILWALID;
}

/*!
 * @brief  Relwrsive called function to print out the stack trace
 *
 * @param[in]  stackContent    Pointer to the content of the whole stack
 * @param[in]  stackSize       The size of the stack
 * @param[in]  stackIdx        The position of the stack we have lwrrently parsed at.
 * @param[in]  lwrPC           The current PC we have parsed at.
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 * @param[in]  frameIdx        Keeps the frame level (starts from 0 and goes up)
 * @param[in]  numNotCalls     Keeps how many functions we have parsed that are not
                               go from call inst (if larger than 2, give a warning)
 */
void 
pmustStacktraceForTasks
(
    LwU8       *stackContent,
    LwU32       contentSize,
    LwU32       stackSize,
    LwS32       stackIdx,
    LwU32       lwrPC,
    LwU32       ucodeVersion
)
{
    if (!bPmuFuncLoaded)
    {
        dprintf("lw: You can use pmust -l [objdumpFile] to load the objdump file.\n");
        dprintf("lw: Now LwWatch will try to load it from lwsym.\n");
        if (!pmustLoad(NULL, ucodeVersion, FALSE))
        {
            dprintf("lw: Unable to load the objdump file from lwsym.\n");
            dprintf("lw: You need to load the objdump file by providing the path to the file\n");
            return;
        }
        else
            dprintf("lw: objdump file successfully loaded from lwsym.\n");
    }

    printStacktraceHelper(stackContent, contentSize, stackSize, 0, lwrPC, ppPmuFuncTable, NumPmuFuncs, 0, 0);
}

/*!
 * @brief  Relwrsive called function to print out the stack trace
 *
 * @param[in]  stackContent    Pointer to the content of the whole stack
 * @param[in]  stackSize       The size of the stack
 * @param[in]  stackIdx        The position of the stack we have lwrrently parsed at.
 * @param[in]  lwrPC           The current PC we have parsed at.
 * @param[in]  ppPmuFuncTable  Function array containing all functions
 * @param[in]  numPmuFuncs     Total number of functions
 * @param[in]  frameIdx        Keeps the frame level (starts from 0 and goes up)
 * @param[in]  numNotCalls     Keeps how many functions we have parsed that are not
                               go from call inst (if larger than 2, give a warning)
 */
static void
printStacktraceHelper
(
    LwU8       *stackContent,
    LwU32       contentSize,
    LwU32       stackSize,
    LwS32       stackIdx,
    LwU32       lwrPC,
    PmuFunc   **ppPmuFuncTable,
    LwU32       numPmuFuncs,
    LwU32       frameIdx,
    LwU32       numNotCalls
)
{
    PmuFunc   *pLwrFunc = pmuFuncSearch(ppPmuFuncTable, numPmuFuncs, lwrPC, FALSE);
    LwU32      offset;
    LwU32      index;
    LwF32      stackUsage;

    if (pLwrFunc == NULL)
    {
        dprintf("Unable to find a function containing pc 0x%x, possibly the stack is corrupted\n", lwrPC);
        return;
    }

    for (index = 0; (index < pLwrFunc->numStackOps) && (pLwrFunc->pStackOps[index].addr < lwrPC); ++index)
    {
        switch (pLwrFunc->pStackOps[index].opType)
        {
        case STACK_OP_PUSH:
            stackIdx += 4;
            break;
        case STACK_OP_POP:
            break;
        case STACK_OP_PUSHM:
            stackIdx += pLwrFunc->pStackOps[index].value;
            break;
        case STACK_OP_POPM:
            break;
        case STACK_OP_ADDSP:
            if (pLwrFunc->pStackOps[index].value < 0)
                stackIdx -= pLwrFunc->pStackOps[index].value;
            break;
        }
    }

    offset = lwrPC - pLwrFunc->entryPoint;
    stackUsage = (LwF32)(contentSize - stackIdx) / (LwF32)stackSize * 100.0f;

    if (stackIdx < 0 || (LwU32)stackIdx > contentSize)
    {
        dprintf("#%-3d   %4.1f%%   0x%05x  at  %s + 0x%05x\n", frameIdx, stackUsage, lwrPC, pLwrFunc->funcName, offset);
        dprintf("Unable to find the function who called %s, possibliy the stack is corrupted\n", pLwrFunc->funcName);
        return;
    }
    else if ((LwU32)stackIdx == contentSize)
    {
        dprintf("#%-3d   %4.1f%%   0x%05x  at  %s + 0x%05x\n", frameIdx, stackUsage, lwrPC, pLwrFunc->funcName, offset);
        return;
    }
    else
    {
        LwU32 prevPC = *(LwU32 *)(&stackContent[stackIdx]);
        CALL_TYPE res = validateCallRelation(pLwrFunc, prevPC, ppPmuFuncTable, numPmuFuncs);
        stackIdx += 4;

        if (res == CALL_TYPE_VALID)
            dprintf("#%-3d   %4.1f%%   0x%05x  at  %s + 0x%05x\n", frameIdx, stackUsage, lwrPC, pLwrFunc->funcName, offset);
        else if (res == CALL_TYPE_CALLFROMREG)
            dprintf("#%-3d   %4.1f%%   0x%05x  at  %s + 0x%05x through register\n", frameIdx, stackUsage, lwrPC, pLwrFunc->funcName, offset);
        else
        {
            dprintf("#%-3d   %4.1f%%   0x%05x  at  %s + 0x%05x (!!Not through a \"call\") \n",
                    frameIdx,
                    stackUsage,
                    lwrPC,
                    pLwrFunc->funcName,
                    offset);

            ++numNotCalls;
            if (numNotCalls >= 4)
            {
                dprintf("\n It seems the stack is corrupted, we stop printing out stack trace.\n");
                return;
            }
        }

        if ((LwU32)stackIdx <= contentSize)
            printStacktraceHelper(stackContent, contentSize, stackSize, stackIdx, prevPC, ppPmuFuncTable, numPmuFuncs, frameIdx + 1, numNotCalls);
    }
}

/*!
 * @brief  print out the stack trace
 *
 * @param[in]  topOfStack      Pointer to the top of the stack
 * @param[in]  stackSize       The size of the stack
 * @param[in]  ucodeVersion    ucode version, give an error if mismatch on board and objdump
 */
void
pmustPrintStacktrace 
(
    LwU32 topOfStack,
    LwU32 stackSize,
    LwU32 ucodeVersion
)
{
    const char *symStackStart  = "_isr_stack_start";
    const char *symStackEnd    = "_isr_stack_end";
    const char *symLwrrentTCB  = "pxLwrrentTCB";
    const LwU32 MAX_STACK_SIZE = 4096;
    // a0-a15, CSW,    uxCriticalNesting
    const LwU32 numRegsStored  = 18;
    LwU32       pc             = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_PC);
    LwU32       sp             = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_SP);
    LwU32       port           = 0x1;

    PMU_SYM    *pMatches;
    BOOL        bExactFound;
    BOOL        bValid;
    LwU32       count;
    LwU32       symValueStackStart;
    LwU32       symValueStackEnd;
    LwU32       symValueLwrrentTCB;
    LwU32       realStackUsedSize;
    LwU32       alignedVersion;
    LwU8       *pStack = NULL;
    LwU32       wordsRead;

    if (!bPmuFuncLoaded)
    {
        dprintf("lw: You can use pmust -l [objdumpFile] to load the objdump file.\n");
        dprintf("lw: Now LwWatch will try to load it from lwsym.\n");
        if (!pmustLoad(NULL, ucodeVersion, FALSE))
        {
            dprintf("lw: Unable to load the objdump file from lwsym.\n");
            dprintf("lw: You need to load the objdump file by providing the path to the file\n");
            return;
        }
        else
            dprintf("lw: objdump file successfully loaded from lwsym.\n");
    }

    // Firstly, we need to check if sp is in ISR.
    pMatches = pmuSymFind(symStackStart, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        symValueStackStart = pMatches->addr;
    }
    else
    {
        dprintf("LwWatch cannot find the symbol \"%s\" in .nm file, the code is possibly required to be updated.\n", symStackStart);
        return;
    }
    
    pMatches = pmuSymFind(symStackEnd, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        symValueStackEnd = pMatches->addr;
    }
    else
    {
        dprintf("LwWatch cannot find the symbol \"%s\" in .nm file, the code is possibly required to be updated.\n", symStackEnd);
        return;
    }

    if ((sp >= symValueStackStart) && (sp < symValueStackEnd))
    {
        // In the ISR handling program, so we need to firstly print out the stack trace of the ISR.
        realStackUsedSize = symValueStackEnd - sp;
        alignedVersion = (realStackUsedSize + sizeof(LwU32) - 1) / sizeof(LwU32);
        pStack = (LwU8 *)malloc(alignedVersion << 2);
        pPmu[indexGpu].pmuDmemRead(sp,
                                   LW_TRUE,
                                   alignedVersion,
                                   port,
                                   (LwU32 *)pStack);
        dprintf("Stack trace for ISR routine:\n");

        printStacktraceHelper(pStack, realStackUsedSize, (stackSize << 2), 0, pc, ppPmuFuncTable, NumPmuFuncs, 0, 0);
        dprintf("\nStack trace for task before ISR:\n");
        free(pStack); 
        pStack = NULL;

        // Then we need to print out the stack trace of the original task
        pMatches = pmuSymFind(symLwrrentTCB, FALSE, &bExactFound, &count);
        bValid = FALSE;
        if (bExactFound)
            if (PMU_DEREF_DMEM_PTR(pMatches->addr, port, &symValueLwrrentTCB))
                if (PMU_DEREF_DMEM_PTR(symValueLwrrentTCB, port, &symValueLwrrentTCB))
                    bValid = TRUE;

        if (!bValid)
        {
            dprintf("LwWatch cannot find the symbol \"%s\" in .nm file, the code is possibly required to be updated.\n", symLwrrentTCB);
            return;
        }

        sp = symValueLwrrentTCB + numRegsStored * sizeof(LwU32);
        // Recovered the original PC
        wordsRead = pPmu[indexGpu].pmuDmemRead(sp,
                                               LW_TRUE,
                                               sizeof(pc) >> 2,
                                               port,
                                               (LwU32 *)&pc);
        if (wordsRead != (sizeof(pc) >> 2))
        {
            dprintf("ERROR: Unable to read PC from stack at addr 0x%x\n", sp);
            return;
        }
        // Recovered the original SP
        sp += sizeof(LwU32);
    }


    if ((stackSize == 0) || ((stackSize << 2) > MAX_STACK_SIZE))
    {
        realStackUsedSize = MAX_STACK_SIZE;
        alignedVersion = MAX_STACK_SIZE;
    }
    else
    {
        if ((topOfStack > sp) || ((topOfStack + stackSize * 4) <= sp))
        {
            dprintf("ERROR:  STACK OVERFLOW! Parse the stacktrace as much as it can.\n");
        }

        if ((topOfStack + stackSize * 4) <= sp)
        {
            realStackUsedSize = MAX_STACK_SIZE;
        }
        else
        {
            realStackUsedSize = topOfStack + stackSize * 4 - sp;
        }
        alignedVersion = (realStackUsedSize + sizeof(LwU32) - 1) / sizeof(LwU32);
    }

    pStack = (LwU8 *)malloc(alignedVersion << 2);

    wordsRead = pPmu[indexGpu].pmuDmemRead(sp,
                                           LW_TRUE,
                                           alignedVersion,
                                           port,
                                           (LwU32 *)pStack);

    if (wordsRead != alignedVersion)
    {
        dprintf("ERROR: Unable to read stack at addr 0x%x\n", sp);
    }
    else
    {
        printStacktraceHelper(pStack, realStackUsedSize, (stackSize << 2), 0, pc, ppPmuFuncTable, NumPmuFuncs, 0, 0);
    }
    free(pStack);
    dprintf("\nStack Trace Finished.\n");
}
