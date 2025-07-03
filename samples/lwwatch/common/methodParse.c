/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 6.30.2005
// methodParse.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "print.h"
#include "chip.h"
#include "methodParse.h"
#include "hwref/lwclass.h"

//
// statics
//
static void _getSubValsFromName(char * regName, LwU32 regValue, FILE *pfile, BOOL isMultiLinePrint);
static BOOL _parseAndRankMethod(char *methodArgs, LwU32 methodAddr, LwS32 *lastBitsCommon, BOOL *isMulBitsCommon);
static BOOL _getLineArgs(char *lwrLine, char *outArg1, char *outArg2, LwU32 argLen, FILE *pfile);
static BOOL _isMethodSubvalue(char *methName);
static FILE * _openClassHeader(LwU32 classNum);
static void _printAlignmentSpaces(char *str);
static BOOL _isMemberClassBlockList(LwU32 classNum, char *method);
static BOOL _isMemberManualBlockList(char *method);
static BOOL _createManualTable(ManualTableElement_t *manTable, char *manList[], LwU32 manTableLen);
static BOOL _getNextManualMemSpan(FILE *pFile, BoundListElement_t *bound);
static FILE * _openManualFile(char *manName);
static BOOL _isValidManualMemSpan(FILE *pFile, char *spanName);
static char *GetChipManualDir(LwU32 chipInfo);

static char *manual_dir = NULL;

//-----------------------------------------------------
// parseClassHeader
// + This will open a header file for classNum, attempt to find
// + a match to the method in it, and parse the data to the matched method.
// + return TRUE if the method and data were found, parsed, and printed.
// + return FALSE if nothing was found and printed.
//-----------------------------------------------------
BOOL parseClassHeader(LwU32 classnum, LwU32 method, LwU32 data)
{
    FILE *pfile;
    char lwrLine[MAX_STR_LEN], argName[MAX_STR_LEN], argValue[MAX_STR_LEN];
    BOOL isParsed = FALSE, isMulMethBitsCom = FALSE;
    long lastFpos = 0, methBitsComFpos = 0;
    LwS32 methBitsCom = -1, idx;
    LwU32 parsedMeth;
    char *ptr;
    HostMethods_t hostMethods[] = HOST_METHODS;
    LwS32 len = sizeof(hostMethods)/sizeof(hostMethods[0]);

    classnum &= 0xffff;

    // Colwert the HW class number to the SW class if applicable
    switch (classnum)
    {
        case LW30_RANKINE_PRIMITIVE_HW_CLASSNUM:
        {
            classnum = 0x3097;
            break;
        }

        case LW34_RANKINE_PRIMITIVE_HW_CLASSNUM:
        {
            classnum = 0x3497;
            break;
        }

        case LW35_RANKINE_PRIMITIVE_HW_CLASSNUM:
        {
            classnum = 0x3597;
            break;
        }

        default:
            break;
    }

    if (((classnum & 0xff) != 0x7c) &&
        ((classnum & 0xff) != 0x7d) &&
        ((classnum & 0xff) != 0x7e))
    {
        //
        // Check the methods that exist for all classes like SET_OBJECT.
        // No parsing necessary to determine the value of these HOST_METHODS.
        //
        for (idx = 0; method < 0x100 && idx < len; idx++)
        {
            if (method == hostMethods[idx].methodNum)
            {
                dprintf("lw: 0x%04x: %s: ", method, hostMethods[idx].methodName);

                // Print some spaces to align data left
                _printAlignmentSpaces(hostMethods[idx].methodName);
                dprintf("0x%08x", data);
                return TRUE;
            }
        }
    }

    // Open the class header file
    if ((pfile = _openClassHeader(classnum)) == NULL)
    {
        static int printedOnce = 0;

        if (printedOnce)
        {
            dprintf("lw: %s - cannot open class header file\n", __FUNCTION__);
            dprintf("lw: Check elwironmental variable LWW_CLASS_SDK\n");
            printedOnce = 1;
        }

        return FALSE;
    }

    // Try to find an exact match to the method number, stop and print on the first match
    while (fgets(lwrLine, MAX_STR_LEN, pfile) != NULL)
    {
        // Succeeds if of the form #define NAME VALUE, the NAME is not a
        // subvalue of the last register, and the NAME is not on the Block List
        if (_getLineArgs(lwrLine, argName, argValue, MAX_STR_LEN, pfile) &&
            !_isMethodSubvalue(argName) &&
            !_isMemberClassBlockList(classnum, argName))
        {
            // Remove the beginning braces of method data
            for (ptr = argValue; *ptr == '('; ptr++)
                ;

            // Succeeds if the method value is an exact match, stop searching and start parsing.
            if (sscanf(ptr, "%x", &parsedMeth) && (parsedMeth == method))
            {
                // We have found a match, Now we need to parse and print the data
                dprintf("lw: 0x%04x: %s: ", method, argName);
                _getSubValsFromName(argName, data, pfile, FALSE);
                isParsed = TRUE;
                break;
            }
            // This NAME is a function, we need to see if our method could possibly be this one and
            // keep track of the best match (in case we don't find an exact match).
            else if (strchr(argName, '('))
            {
                // This will compare methods to see if we have a possible match of
                // the form (0x003+(i)*4+(j)*8+(k)*16)
                if (_parseAndRankMethod(argValue, method, &methBitsCom, &isMulMethBitsCom))
                {
                    methBitsComFpos = lastFpos;
                }
            }
        }

        // Keeps track of last file position in case this method is a function.
        // This way we can jump directly to our best match.
        lastFpos =  ftell(pfile);
    }

    // If there was no exact match to the method,
    // We could be a multiple of some methods i.e. (0x0030+(i)*4)
    // Use our best match from the search above.
    if (!isParsed && (methBitsCom != -1) && !fseek(pfile, methBitsComFpos, 0))
    {
        // This happens when two or more functions have the exact same probability
        // of being the correct function
        if (isMulMethBitsCom)
        {
            dprintf("lw: WARNING: The following parsed method is equally common with another method.\n");
        }

        if ((fgets(lwrLine, MAX_STR_LEN, pfile) != NULL) &&
            _getLineArgs(lwrLine, argName, argValue, MAX_STR_LEN, pfile))
        {
            dprintf("lw: 0x%04x: %s: ", method, argName);
            _getSubValsFromName(argName, data, pfile, FALSE);
            isParsed = TRUE;
        }
    }

    fclose(pfile);

    return isParsed;
}

//-----------------------------------------------------
// _getLineArgs
// + Parse line of form: #define NAME VALUE
// + return TRUE if successfully parsed and populated outArg1 and outArg2
//-----------------------------------------------------
static BOOL _getLineArgs(char *lwrLine, char *outArg1, char *outArg2, LwU32 argLen, FILE *pfile)
{
    LwU32 lineIndex;
    LwU32 argIndex, numOpenParens;
    *outArg1 = *outArg2 = 0;

    // Remove spaces
    for(lineIndex = 0; lwrLine[lineIndex] == ' ' || lwrLine[lineIndex] == '\t'; lineIndex++);
    if (lwrLine[lineIndex] == 0)
    {
        return FALSE;
    }

    // Line must begin with #define, otherwise we don't care
    if (strncmp(lwrLine+lineIndex, "#define", 7))
    {
        return FALSE;
    }

    // Remove spaces
    for(lineIndex += 7; lwrLine[lineIndex] == ' ' || lwrLine[lineIndex] == '\t'; lineIndex++);
    if (lwrLine[lineIndex] == 0)
    {
        return FALSE;
    }

    // Remove #define name and store in outArg1
    numOpenParens = 0;
    for(argIndex = 0; ((lwrLine[lineIndex] != ' ' || lwrLine[lineIndex] == '\t') &&
                       !numOpenParens && lwrLine[lineIndex] != 0) ||
                      (numOpenParens && lwrLine[lineIndex] != 0); lineIndex++)
    {
        // This Name is a function, we don't want to end on a space
        // unless all of the parens are closed
        if (lwrLine[lineIndex] == '(')
        {
            numOpenParens++;
        }
        else if (lwrLine[lineIndex] == ')')
        {
            numOpenParens--;
        }

        // If the length of the name is greater than the buffer size, do nothing
        // Maybe should add error, or just directly modify input string
        if ((argLen-1) > argIndex)
        {
            outArg1[argIndex++] = lwrLine[lineIndex];
        }
    }
    outArg1[argIndex] = 0;

    // The last char we read was NULL, return FALSE because parsing is not done
    if (lwrLine[lineIndex] == 0)
    {
        *outArg1 = 0;
        return FALSE;
    }

    // Remove spaces
    for(; lwrLine[lineIndex] == ' ' || lwrLine[lineIndex] == '\t'; lineIndex++);
    if (lwrLine[lineIndex] == 0)
    {
        return FALSE;
    }

    // The rest of the characters are part of the value
    for(argIndex = 0; lwrLine[lineIndex] != 0; lineIndex++)
    {
        // The value could span multiple lines if \ is present, we need to combine the lines.
        // May need to check that \ is not within quotes (single and double).
        if (lwrLine[lineIndex] == '\\')
        {
            if (fgets(lwrLine, MAX_STR_LEN, pfile) == NULL)
            {
                // I don't know if this is an error.  We have a multi-line define
                // but we cannot get the next line.
                break;
            }
            lineIndex = (-1);
        }
        // If the character is not whitespace, add it to outArg2
        else if (lwrLine[lineIndex] != ' ' &&
            lwrLine[lineIndex] != '\t' &&
            lwrLine[lineIndex] != '\n' &&
            lwrLine[lineIndex] != '\r' &&
            (argLen-1) > argIndex)
        {
            outArg2[argIndex++] = lwrLine[lineIndex];
        }
    }
    outArg2[argIndex] = 0;
    return TRUE;
}



//-----------------------------------------------------
// _getSubValsFromName
// + Assumes the Form
// + regname
// + regname_XXX     I:J
// + regname_XXX_YY  0xZZZZ
// + Also, the subvalues must be right after the register.
//-----------------------------------------------------
static void _getSubValsFromName(char * regName, LwU32 regValue, FILE *pfile, BOOL isMultiLinePrint)
{
    char lwrLine[MAX_STR_LEN], argName[MAX_STR_LEN], argValue[MAX_STR_LEN];
    LwU32 subRegVal = regValue, lwrVal, upperBound, lowerBound;
    BOOL dataValPrinted = TRUE, isFirstVal = TRUE, isFirstValForField = TRUE, skippedBaseName = FALSE;
    char *ptr;
    size_t regNameLen;
    size_t subValNameLen = 0;

    if (!isMultiLinePrint)
    {
        // Print some spaces to align data left
        _printAlignmentSpaces(regName);
    }

    // If this method has braces we have to remove them
    // so the subvalue name comparisons will work
    ptr = strchr(regName, '(');
    if (ptr != NULL)
    {
        *ptr = 0;
    }
    regNameLen = strlen(regName);

    // We assume that all subparts of this method have the same name base
    // Parse until we find a different name base.
    while (fgets(lwrLine, MAX_STR_LEN, pfile) != NULL)
    {
        // This could be a new line or comments, don't stop until we are sure
        // the next argName does not have the same base.S
        if (!_getLineArgs(lwrLine, argName, argValue, MAX_STR_LEN, pfile))
        {
            continue;
        }

        // Bad name base is found, just incase we are out of order
        if (strncmp(argName, regName, regNameLen))
        {
            skippedBaseName = TRUE;
            continue;
        }

        // These dword register indicators are in manual files.
        if (strstr(argValue, "4R*/") || strstr(argValue, "4A*/") || strchr(argName, '('))
        {
            break;
        }

        // Remove the beginning braces of method data
        for(ptr = argValue; *ptr == '('; ptr++);

        // This is a bit selection subvalue
        if (sscanf(ptr, "%u:%u", &upperBound, &lowerBound) == 2)
        {
            skippedBaseName = FALSE;

            // If the data from the last subvalue wasn't printed, print it.
            if (!dataValPrinted)
            {
                dprintf("0x%x", subRegVal);
            }

            // Separates the method subvalues
            if (!isFirstVal)
            {
                if (!isMultiLinePrint)
                {
                    dprintf(", ");
                }
                else
                {
                    dprintf("\n");
                }
            }

            // Print only the part of the subvalue name that differs from the method name.
            // Add 1 to remove the leading '_'
            if (!isMultiLinePrint)
            {
                dprintf("%s = ", argName+regNameLen+1);
            }
            else
            {
                dprintf("lw:\t%s", argName+regNameLen+1);
                _printAlignmentSpaces(argName+regNameLen+1);
                dprintf(" = ");
            }
            subValNameLen = strlen(argName) - regNameLen;

            dataValPrinted = FALSE;
            isFirstVal = FALSE;
            isFirstValForField = TRUE;

            // We cannot call a DRF_VAL() since it is a #define that operates on the subvalue name.
            // This is really a DRF_VAL() of upperBound:lowerBound on regValue.
            subRegVal = (regValue>>(lowerBound % 32))&(0xFFFFFFFF>>(31-(upperBound%32)+(lowerBound%32)));
        }
        // This is a specific value for the subvalue.
        else if (sscanf(ptr, "%x", &lwrVal))
        {
            if (skippedBaseName)
            {
                break;
            }

            // The specific value and the current subvalue match, then print out the
            // the part of the specific subvalue name that differs from the subvalue name.
            // Add 1 to remove the leading '_'
            if (subRegVal == lwrVal)
            {
                if (isFirstVal && isMultiLinePrint)
                {
                    dprintf("lw:\tVALUE");
                    _printAlignmentSpaces("VALUE");
                    dprintf(" = ");
                }
                if (isFirstValForField == FALSE)
                {
                    dprintf(", ");
                }

                dprintf("%s", argName+regNameLen+subValNameLen+1);
                dataValPrinted = TRUE;
                isFirstVal = FALSE;
                isFirstValForField = FALSE;
            }
        }
        else
        {
            //
            // This case can occur for a register that has a string for a value
            // Just print the string value
            //

            // Remove the ending comment
            for(ptr = argValue; *ptr != '/' && *ptr != 0; ptr++);

            *ptr = 0;

            if (!isMultiLinePrint)
            {
                dprintf("%s", argName+regNameLen+subValNameLen+1);
            }
            else
            {
                dprintf("lw:\t%s", argName+regNameLen+subValNameLen+1);
                _printAlignmentSpaces(argName+regNameLen+subValNameLen+1);
            }

            dprintf(" = %s", argValue);
            dataValPrinted = TRUE;
            isFirstVal = FALSE;
            isFirstValForField = TRUE;
        }
    }

    // Print the any remaining data
    if (!dataValPrinted)
    {
        dprintf("0x%x", subRegVal);
    }

    // If nothing was parsed we still want to print the data
    if (isFirstVal && !isMultiLinePrint)
    {
        dprintf("0x%08x", regValue);
    }

    if (isMultiLinePrint)
    {
        dprintf("\n");
    }
}

//-----------------------------------------------------
// _parseAndRankMethod
// + Compares the given methodArgs to the methodAddr to see if we have a possible match
// + return TRUE if lastBitsCommon and isMulBitsCommon are modified i.e. we
//   need to update the file position.
//-----------------------------------------------------
static BOOL _parseAndRankMethod(char *methodArgs, LwU32 methodAddr,
                                LwS32 *lastBitsCommon, BOOL *isMulBitsCommon)
{
    LwU32 regAddr, regIncr;
    LwS32 lwrBitsCommon = 0, i;
    BOOL isChangedVal = FALSE;
    BOOL isPossibleMultiple = FALSE;
    char *ptr, tmpC;

    // Remove the beginning braces
    for (ptr = methodArgs; *ptr == '('; ptr++)
        ;

    // Get base address
    if (!sscanf(ptr, "%x", &regAddr))
    {
        return FALSE;
    }

    // Find the multiples of this value. For example, (0x00400+(i)*8+(j)*4)
    // and we want to know if 0x412 is a possible value. This is useful when we
    // have (0x00400+(i)*8) and (0x00404+(i)*8) as methods.
    while (!isPossibleMultiple && (ptr = strchr(ptr+1, '+')) &&
           ((sscanf(ptr, "+(%c)*%u", &tmpC, &regIncr) == 2 && regIncr != 0) ||
            sscanf(ptr, "+(%c)*%x", &tmpC, &regIncr) == 2))
    {
        if ((i = (methodAddr - regAddr)) >= 0)
        {
            isPossibleMultiple = !(i % regIncr);
        }
    }

    if (isPossibleMultiple)
    {
        // Starting at the MSB, see how many bit are in common
        for(i=31; i>=0 && (((regAddr>>i) & 0x1) == ((methodAddr>>i) & 0x1)); i--)
        {
            lwrBitsCommon++;
        }

        // This method is a better match than the last one, update our data
        if (*lastBitsCommon < lwrBitsCommon)
        {
            *lastBitsCommon = lwrBitsCommon;
            *isMulBitsCommon = FALSE;
            isChangedVal = TRUE;
        }
        else if (*lastBitsCommon == lwrBitsCommon)
        {
            // There are two equally possible values.
            // This means we could be wrong and we need to keep track of this.
            *isMulBitsCommon = TRUE;
        }
    }

    return isChangedVal;
}

//-----------------------------------------------------
// _isMethodSubvalue
// + When matching data to header file methods, we want to make sure we do not
// + match a subvalue. Since this is not a valid value.
// + return TRUE if the methName is a subvalue
//-----------------------------------------------------
static BOOL _isMethodSubvalue(char *methName)
{
    static char   lastMethName[MAX_STR_LEN];
    static size_t lastMethNameLen = 0;
    char *ptr = NULL;
    LwU32 isSameNameBase;

    // This methName is a function, thus it must not be a subvalue
    ptr = strchr(methName, '(');
    isSameNameBase = !strncmp(methName, lastMethName, lastMethNameLen);

    // If this is not a subvalue of the register,
    // i.e. is function or is first run or different name base
    if (ptr != NULL || !lastMethNameLen || !isSameNameBase)
    {
        // Store this as the lastMethName
        strcpy(lastMethName, methName);

        // If this is a function, remove the braces so the
        // name base compare will work.
        if (ptr != NULL)
        {
            ptr = strchr(lastMethName, '(');
            *ptr = 0;
        }

        lastMethNameLen = strlen(lastMethName);
        return FALSE;
    }
    // Special case where the same method appears twice, i.e. across two diff searches
    else if (isSameNameBase && (strlen(methName) == lastMethNameLen))
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

//-----------------------------------------------------
// _openClassHeader
// + Opens a class header for the given classNum
// + Will first check LWW_CLASS_SDK elwironmental var for path
// + Next will go out to network, defined in CLASS_PATH_SERVER
// + return a pointer to the open class header. NULL if it fails.
//-----------------------------------------------------
static FILE * _openClassHeader(LwU32 classNum)
{
    char fName[MAX_STR_LEN];
    char *ptr;
    FILE *pFile = NULL;

    // Check elwironmental variable for path, i.e. c:\sw\dev\gpu_drv\chips_a
    // Open the file locally to be parsed
    if ((ptr = getelw("LWW_CLASS_SDK")) != NULL)
    {
        // Some people add the trailing \ to the path
        if (ptr[strlen(ptr)-1] == DIR_SLASH_CHAR)
        {
            sprintf(fName, "%s" CLASS_PATH_LOCAL, ptr, classNum);
        }
        else
        {
            sprintf(fName, "%s" DIR_SLASH CLASS_PATH_LOCAL, ptr, classNum);
        }

        pFile = fopen(fName, "r");
    }

    // Open the file on the server to be parsed
    if (pFile == NULL)
    {
        sprintf(fName, CLASS_PATH_SERVER, classNum);
        pFile = fopen(fName, "r");
    }

    return pFile;
}

//-----------------------------------------------------
// isValidClassHeader
// + return TRUE if there is a class header for a given classNum
//-----------------------------------------------------
BOOL isValidClassHeader(LwU32 classNum)
{
    FILE * pFile;

    // Right now we check if we can open the class header.
    // Then we close the header if possible, but this seems inefficient.
    pFile = _openClassHeader(classNum);

    if (pFile == NULL)
    {
        return FALSE;
    }
    else
    {
        fclose(pFile);
        return TRUE;
    }
}

//-----------------------------------------------------
// _isMemberClassBlockList
// + This will block methods from being selected
// + since some are in every class but are unwanted i.e. LWXXX_NOTIFIERS_*
// + return TRUE if the method is a member of the Block List
//-----------------------------------------------------
static BOOL _isMemberClassBlockList(LwU32 classNum, char *method)
{
    char *blockList[] = BLOCK_LIST_CLASS;
    char lwrBlockStr[MAX_STR_LEN];
    LwU32 i;

    // Can do sizeof since the array size is defined at compile time.
    for (i = 0; i < sizeof(blockList)/sizeof(char*); i++)
    {
        // This will paste the classNum into the blockList string
        // NOTE: If the string has no parameters, this will not cause an error.
        sprintf(lwrBlockStr, blockList[i], classNum);
        if (!strncmp(method, lwrBlockStr, strlen(lwrBlockStr)))
        {
            return TRUE;
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// _isMemberManualBlockList
// + This will block methods from being selected
// + return TRUE if the method is a member of the Block List
//-----------------------------------------------------
static BOOL _isMemberManualBlockList(char *method)
{
    char *blockList[] = BLOCK_LIST_MANUAL;
    LwU32 i;

    // Can do sizeof since the array size is defined at compile time.
    for (i = 0; i < sizeof(blockList)/sizeof(char*); i++)
    {
        if (!strncmp(method, blockList[i], strlen(blockList[i])))
        {
            return TRUE;
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// _printAlignmentSpaces
// + Print some spaces to align data left
//-----------------------------------------------------
static void _printAlignmentSpaces(char *str)
{
    LwS32 i;
    LwU8 lwrLine[MAX_STR_LEN];

    i = ALIGNMENT_SPACES - (LwS32)strlen(str);
    if (i < 0)
    {
        i = 0;
    }
    else if (i >= MAX_STR_LEN)
    {
        i = MAX_STR_LEN - 1;
    }

    memset(lwrLine, ' ', i);
    lwrLine[i] = 0;
    dprintf("%s", lwrLine);
}

//-----------------------------------------------------
// _getNextManualMemSpan
// + This will parse out one memory range for the manual file
// + FALSE if there are no more ranges to be parsed.
//-----------------------------------------------------
static BOOL _getNextManualMemSpan(FILE *pFile, BoundListElement_t *bound)
{
    char lwrLine[MAX_STR_LEN], argName[MAX_STR_LEN], argValue[MAX_STR_LEN];
    LwU32 upperBound, lowerBound;
    char *ptr;

    while (fgets(lwrLine, MAX_STR_LEN, pFile) != NULL)
    {
        // Some memspace spans are commented out in header files
        // Remove spaces
        for(ptr = lwrLine; *ptr == ' '  || *ptr == '\t'; ptr++);
        if (*ptr == 0) continue;

        // Remove comment
        for(; *ptr == '/'; ptr++);
        if (*ptr == 0) continue;

        // Succeeds if of the form #define NAME VALUE
        if (_getLineArgs(ptr, argName, argValue, MAX_STR_LEN, pFile))
        {
            // Remove the beginning braces of method data
            for(ptr = argValue; *ptr == '('; ptr++);

            // This is a bit selection subvalue
            if (sscanf(ptr,"%x:%x", &upperBound, &lowerBound) == 2)
            {
                bound->high = upperBound;
                bound->low = lowerBound;
                if (_isValidManualMemSpan(pFile, argName))
                {
                    return TRUE;
                }
            }
            else
            {
                return FALSE;
            }
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// _isValidManualMemSpan
// + return TRUE if the given spanName is valid for the fiven pFile
//-----------------------------------------------------
static BOOL _isValidManualMemSpan(FILE *pFile, char *spanName)
{
    char   lwrLine[MAX_STR_LEN], argName[MAX_STR_LEN], argValue[MAX_STR_LEN];
    size_t spanNameLen;
    BOOL   isValid = FALSE;
    long   startPos;

    // Is this on the block list?
    if (_isMemberManualBlockList(spanName))
    {
        return FALSE;
    }

    // We want to go back to the last position of the file when we are done
    startPos = ftell(pFile);
    spanNameLen = strlen(spanName);

    while (fgets(lwrLine, MAX_STR_LEN, pFile) != NULL)
    {
        // Succeeds if of the form #define NAME VALUE,
        // spanName is the base of a DWORD register,
        // and has register indicator commments
        if (_getLineArgs(lwrLine, argName, argValue, MAX_STR_LEN, pFile) &&
            !strncmp(argName, spanName, spanNameLen) &&
            (strstr(argValue, "4R*/") || strstr(argValue, "4A*/")))
        {
            isValid = TRUE;
            break;
        }
    }

    fseek(pFile, startPos, 0);
    return isValid;
}

//-----------------------------------------------------
// _openManualFile
// + Open the manual file
// + Return NULL if the operation fails, otherwise return a pointer to the file
// Nov07 - dropped support for //hw manuals.  Was not looking at compiled manuals.
// sw manuals should be complete for >= lw50 chips
//-----------------------------------------------------
static FILE * _openManualFile(char *manName)
{
    char fileName[MAX_STR_LEN];
    LwU32 chipInfo, revInfo;
    char *chipDir;
    FILE *pFile;

    // Get the path to the manuals from the elwironmental variable LWW_MANUAL_SDK
    if (manual_dir == NULL)
    {
        return NULL;
    }

    // What kind of chip are we running i.e. LW40
    GetChipAndRevision(&chipInfo, &revInfo);

    chipDir = GetChipManualDir(chipInfo);

    // If the chipInfo is not in the lookup table then generate the directory name
    // Some people add the trailing \ to the path
    if (manual_dir[strlen(manual_dir)-1] == DIR_SLASH_CHAR)
    {
        sprintf(fileName, "%s%s" DIR_SLASH "%s.h", manual_dir, chipDir, manName);
    }
    else
    {
        sprintf(fileName, "%s" DIR_SLASH "%s" DIR_SLASH "%s.h", manual_dir, chipDir, manName);
    }

    pFile = fopen(fileName, "r");

    return pFile;
}

//-----------------------------------------------------
// _createManualTable
// + this will create a list of mem spans for each manual
// + the mem spans are used to determine which range of
// + addresses correspond to which manual
//-----------------------------------------------------
static BOOL _createManualTable(ManualTableElement_t *manTable, char *manList[], LwU32 manTableLen)
{
    FILE *pFile = NULL;
    BOOL isSomeDataSet = FALSE;
    LwU32 i;

    //
    // Initialize the table
    // The table is the same size as the manList and an index into the manTable
    // corresponds to an index in the manList. For example, if manList[22] is "dev_mpeg"
    // then manTable[22] contains the parsed data for "dev_mpeg".
    //
    memset(manTable, 0, manTableLen*sizeof(ManualTableElement_t));
    for (i = 0; i < manTableLen; i++)
    {
        long lastFilePos = 0;

        // Get pointers to the data for easier access
        ManualTableElement_t *tableElement = &manTable[i];
        LwU32 *elements = &tableElement->boundListElements;

        // Open the manual, if we can't then go to next manual
        if ((pFile = _openManualFile(manList[i])) == NULL) continue;

        // Traverse the file as long as there are memory spans
        while((*elements < MAX_LIST_ELEMENTS) &&
              _getNextManualMemSpan(pFile, &tableElement->boundList[*elements]))
        {
            // _getNextManualMemSpan just added to the boundList, increment the number of elements
            (*elements)++;
            // record this file position, so when we come back to parse more registers
            // from this file, we can skip the known bad data
            lastFilePos = ftell(pFile);
            isSomeDataSet = TRUE;

            if (*elements >= MAX_LIST_ELEMENTS)
            {
                dprintf("lw: %s - Bound List is Full, please increase default size of the list.\n", __FUNCTION__);
            }
        }

        // record this file position so everytime we parse this file we start here, skipping the bad data
        tableElement->startFilePos = lastFilePos;
        fclose(pFile);
    }

    return isSomeDataSet;
}

//-----------------------------------------------------
// _parseIReg
// addr: address to be checked
// parsedAddr: parsed address of the start of index range
// data: data at the address
// pFile: file pointer to file lwrrently looking in
// argName: Name of the register
// argValue: Value of the register declaration
// manual: Name of the manual lwrrently looking in
// 
// Parses the indexed register structure; checking to see
// if we can parse it. If everything looks good and it is
// in a format that we understand (n-dimensional array structure)
// then see if the addr is part of it. If it is, print out
// the information and return TRUE, otherwise return FALSE.
//
// Lwrrently, the n in n-dimensional array is limited via
// the MAX_IREG_COUNT. It is lwrrently set to support all 
// lwrrently implemented registers. However, mainly only
// the single 1-dimensional array is of real concern and
// the main intention of this function. 
//-----------------------------------------------------
BOOL _parseIReg(LwU32 addr, LwU32 parsedAddr, LwU32 data, FILE *pFile, char *argName, const char *argValue, const char *manual, BOOL isListAll)
{
    long file_pos = ftell(pFile);
    ireg_t iregs[MAX_IREG_COUNT];
    char line[MAX_STR_LEN];
    char name[MAX_STR_LEN];
    char value[MAX_STR_LEN];
    char *pstr;
    LwU32 iregs_count;
    LwU32 size;
    LwU32 i;
    LwU32 j;
    
    //
    // Parse out the width and character for each index:
    // Note: Code below simply gets rid of a bunch of nested if-statements
    // Line is #define NAME (<parsedAddr>+(i)*<sizeof i>...)
    // Simply parse out all patterns of "+(%C)*%d" until there are no more.
    //
    for (
        pstr = strchr(argValue, '+'), iregs_count = 0;
        pstr != NULL && iregs_count < MAX_IREG_COUNT && 
        (sscanf(pstr, "+(%c)*%d", &(iregs[iregs_count].c), &(iregs[iregs_count].width)) == 2 ||
        sscanf(pstr, "+((%c)*%d)", &(iregs[iregs_count].c), &(iregs[iregs_count].width)) == 2);
        pstr = strchr(pstr + 1, '+'), iregs_count++
        );

    if (0 == iregs_count)
    {
        dprintf("lw: Could not parse register definition.\n");
        return FALSE;
    }
    
    //
    // Parse out the length of each index:
    // While each step succeeds, try to,
    // (1) Read the next line
    // (2) Parse the #define <name> <value>
    // (3) Check if the if we have a <name>__SIZE_<i> definition
    // (4) Parse out the i index and parse out the size value
    //
    while (
        fgets(line, MAX_STR_LEN, pFile) != NULL &&
        _getLineArgs(line, name, value, MAX_STR_LEN, pFile) &&
        (pstr = strstr(name, "__SIZE")) != NULL &&
        sscanf(pstr, "__SIZE_%d", &i) == 1 &&
        (i != 0) && 
        sscanf(value, "%d", &(iregs[i-1].length)) == 1
        );
    
    //
    // Ensure that the indexed register takes form of an n-dimensional array
    // Example: 
    //    #define MY_IREG(i,j)        (0x0+i*12+j*4)
    //    #define MY_IREG__SIZE_1     4              // Rows
    //    #define MY_IREG__SIZE_2     3              // Columns
    //
    // Note that the 12 in the i*12 is implicitly defined by the __SIZE_1 and
    // the 4 in the j*4 part. This is because we assume, in order to determine
    // index values, that the memory layout is a row-major flat array.
    //
    // Thus, each element is 4-bytes in this case. Each row has 3 columns and
    // there are 6 rows. The i multiplier is defined as __SIZE_1 * size element.
    // Thus is has to be 12, otherwise the assumption does not hold and we
    // cannot parse data correctly in the next step.
    //
    // This pattern holds to an n-dimensional array. The (i+1)th index must have
    // a multiplier equal to ith multiplier * __SIZE_i. To check this we start
    // at the end and verify that the ilwariant holds with the "previous" index.
    //
    // ASSUMPTIONS: i,j,k ordering is consistent and decreasing in values (order
    //              shown above). There are very few > 1-d indexed registers 
    //              when checked. Will not work if in this form, but mainly
    //              intended for 1-d indexed registers.
    //
    i = iregs_count - 1;
    size = iregs[i].width * iregs[i].length;
    for (i -= 1, j = 1; j < iregs_count; i--, j++)
    {
        if (iregs[i].width != size)
        {
            return FALSE;
        }
        size = iregs[i].width * iregs[i].length;
    }
    
    //
    // Callwlate the indices of the address:
    // 
    // Based on the assumption above, we simply divide and use the remainder
    // for the next index. Consider the example extended from above.
    //
    // value[1][2] => i=1 and j=2 => 0x0+2*4+1*12 = 0x14
    //
    // i: 0x14 / 12 = 1     0x14 % 12 = 8
    // j: 0x8  /  4 = 2     0x8  %  4 = 0
    //
    // Note that the last value must be 0, otherwise it's not a valid multiple
    // and does not fit in our index register.
    //
    for (i = 0, addr -= parsedAddr; i < iregs_count; i++)
    {
        //
        // Some index registers have a width = 0!!
        // This is mainly seen in 2-d registers and does not make sense, but
        // we must prevent a divide-by-zero.
        //
        if (iregs[i].width == 0)
        {
            if (addr == 0)
            {
                continue;
            }
            else
            {
                return FALSE;
            }
        }

        iregs[i].indexValue = addr / iregs[i].width;
        if (iregs[i].indexValue >= iregs[i].length)
        {
            return FALSE;
        }
        addr %= iregs[i].width;
    }
    if (addr != 0)
    {
        return FALSE;
    }
    
    // Print out the values and register information
    size = (int)(strchr(argName, '(') - argName);
    dprintf("lw: %.*s(", size, argName);
    for (i = 0; i < iregs_count; i++)
    {
        dprintf("%d", iregs[i].indexValue);
        if (i != iregs_count - 1)
        {
            dprintf(",");
        }
    }
    dprintf(") = 0x%08x (%s)\n", data, manual);
    fseek(pFile, file_pos, SEEK_SET);
    _getSubValsFromName(argName, data, pFile, TRUE);
    dprintf("\n");

    // If isListAll is on and this is a 1-D array, list all the entries
    if (isListAll && iregs_count != 1)
    {
        dprintf("lw: \"-a\" only supported for single variable indexed registers.\n");
    }
    else if (isListAll)
    {
        for (j = 0; j < iregs[0].length; j++)
        {
            // Skip the entry that we already printed out
            if (j != iregs[0].indexValue)
            {
                addr = (LwU32)(parsedAddr + iregs[0].width * j);
                data = GPU_REG_RD32(addr);
                dprintf("lw: %.*s(%d) = 0x%08x (at 0x%08x)\n", size, argName, j, data, addr);
                fseek(pFile, file_pos, SEEK_SET);
                _getSubValsFromName(argName, data, pFile, TRUE);
                dprintf("\n");
            }
        }
    }

    return TRUE;
}

#ifndef DRF_WINDOWS
//-----------------------------------------------------
// parseManualReg
// + return TRUE if the addr was successfully parsed from a manual
//-----------------------------------------------------
BOOL parseManualReg(LwU32 addr, LwU32 data, BOOL isListAll)
{
    static char *manList[]   = MANUAL_LIST;
    static ManualTableElement_t manTable[sizeof(manList)/sizeof(char*)];
    static BOOL isFirstRun   = TRUE;
    static LwU32 lastChipInfo = 0;
    char lwrLine[MAX_STR_LEN];
    char argName[MAX_STR_LEN];
    char argValue[MAX_STR_LEN];
    LwU32 manTableLen = sizeof(manList)/sizeof(char*);
    LwU32 lwrChipInfo;
    LwU32 parsedAddr;
    LwU32 revInfo;
    LwU32 i;
    LwU32 j;
    BOOL bFound = FALSE;
    FILE *pFile;
    char *ptr;

    if (manual_dir == NULL)
    {
        if ((manual_dir = getelw("LWW_MANUAL_SDK")) == NULL)
        {
            dprintf("lw: %s - Please set LWW_MANUAL_SDK to a HW manual directory.\n", __FUNCTION__);
            dprintf("lw: For example, " INC_DIR_EXAMPLE "\n");
            return FALSE;
        }
    }

    GetChipAndRevision(&lwrChipInfo, &revInfo);

    //
    // If this is the first run or we have changed chips we need to load
    // the manual table
    //
    if (isFirstRun || (lastChipInfo != lwrChipInfo))
    {
        // Returns FALSE if nothing was set in the table
        isFirstRun = !_createManualTable(manTable, manList, manTableLen);
        if (isFirstRun)
        {
            dprintf("lw: %s - No data found. Please check that LWW_MANUAL_SDK is set to a sw or hw manual directory.\n", __FUNCTION__);
            dprintf("lw: Verify this is a valid directory include %s\n", manual_dir);
            return FALSE;
        }
        lastChipInfo = lwrChipInfo;
    }

    // For each manual
    for (i = 0; i < manTableLen && !bFound; i++)
    {
        //
        // Check the bound list to see if the address passed into
        // this method is in range.  If so, we need to parse this file.
        //
        for (j = 0; (j < manTable[i].boundListElements) &&
                    !(addr >= manTable[i].boundList[j].low &&
                      addr <= manTable[i].boundList[j].high); j++);

        // No match found or File does not exist, go to next file.
        if (j == manTable[i].boundListElements ||
            ((pFile = _openManualFile(manList[i])) == NULL))
        {
            continue;
        }

        // Start parsing file after the mem span elements
        fseek(pFile, manTable[i].startFilePos, 0);

        // Try to find an exact match to the method
        while (!bFound && fgets(lwrLine, MAX_STR_LEN, pFile) != NULL)
        {
            //
            // Succeeds if of the form #define NAME VALUE,
            // these files have comments appended to the value which must be of the register type,
            // and the NAME is not on the Block List
            //
            if (_getLineArgs(lwrLine, argName, argValue, MAX_STR_LEN, pFile) &&
                !_isMemberManualBlockList(argName))
            {
                // Remove the beginning braces of method data
                for (ptr = argValue; *ptr == '('; ptr++);
                
                // Parse out the address
                if (sscanf(ptr, "%x", &parsedAddr) != 1)
                {
                    continue;
                }
                
                // 4R*/ is a normal register entry; look for an exact address match
                if (strstr(argValue, "4R*/") && parsedAddr == addr)
                {
                    // We have found a match, Now we need to parse the data
                    dprintf("lw: %s = 0x%08x (%s)\n", argName, data, manList[i]);
                    _getSubValsFromName(argName, data, pFile, TRUE);
                    dprintf("\n");
                    bFound = TRUE;
                }
                // 4A*/ is an indexed register entry; parse that
                else if (strstr(argValue, "4A*/"))
                {
                    if (_parseIReg(addr, parsedAddr, data, pFile, argName, argValue, manList[i], isListAll))
                    {
                        bFound = TRUE;
                    }
                }
            }
        }

        // Cleanup and close the current manual file
        fclose(pFile);
    }

    // No register found, print some errror message
    if (!bFound)
    {
        dprintf("lw: 0x%08x: 0x%08x\n", lwBar0+addr, data);
        dprintf("lw: Couldn't find register reference \"0x%08x\"\n", addr);
    }

    return bFound;
}
#endif

//
// Colwert a PMC_BOOT0 to an include directory for header files.
// See: //docs/resman/hardware/gpubootid.h
//
static char *
GetChipManualDir(LwU32 chipInfo)
{

    if (IsT124())
    {
        return "t4x/t40";
    }

    switch (chipInfo)
    {
        // lw3x headers are mostly in lw30 with a few in each chip dir
        case 0x30:
        case 0x31:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x38:
            return "lw30";

        case 0x40:
        case 0x45:
            return "lw40";
        case 0x41:
        case 0x42:
            return "lw41";
        case 0x43:
            return "lw43";
        case 0x44:
        case 0x4A:
            return "lw44";
        case 0x46:
            return "lw46";
        case 0x47:
            return "lw47";
        case 0x49:
        case 0x4B:
            return "lw49";
        case 0x4C:
            return "lw4c";
        case 0x4D:
            return "lw4d";
        case 0x4E:
            return "lw4e";
        case 0x63:
            return "lw63";
        case 0x67:
            return "lw67";

        case 0x50:
            return "lw50";
        case 0x84:
            return "g84";
        case 0x86:
            return "g86";
        case 0x92:
            return "g92";
        case 0x94:
            return "g94";
        case 0x96:
            return "g96";
        case 0x98:
            return "gt206";
        case 0x99:
            return "g98";
        case 0xA0:
            return "gt200";
        case 0xA2:
            return "gt212";
        case 0xA3:
            return "gt215";
        case 0xA5:
            return "gt216";
        case 0xA8:
            return "gt218";
        case 0xA6:
        case 0xA7:
        case 0xAA:
        case 0xAB:
            return "gt206";
        case 0xA9:
        case 0xAC:
            return "igt209";
        case 0xAD:
            return "igt21a";
        case 0xAE:
            return "igt21b";

        case 0xC0:
            return "fermi/gf100";
        case 0xC4:
            return "fermi/gf104";
        case 0xC6:
            return "fermi/gf106";
        case 0xC8:
            return "fermi/gf108";

        case 0xE4:
            return "kepler/gk104";
        case 0xE6:
            return "kepler/gk106";
        case 0xE7:
            return "kepler/gk107";
        case 0xEA:
            return "kepler/dt40";
        case 0xF0:
            return "kepler/gk110";

    }

    return "";
}
