/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "stdLocal.h"
#include "ctLog.h"
#include "ptxIR.h"
// OPTIX_HAND_EDIT
#if 0
#include "lwelf_writer.h"
#include "zlib.h"
#include "stdVector.h"
#include "ptxaslibMessageDefs.h" 
// OPTIX_HAND_EDIT
#endif

#include "lwtypes.h"
#include "copi_ucode.h"
#include "copi_inglobals.h"

#include "lwdwarf.h"
#include "dwarf_def.h"
#include "DebugInfoPrivate.h" // for internal maps
#include "DebugInfoToElf.h"
// OPTIX_HAND_EDIT
//#include "leb_decode.h"

#define DEFAULT_ALIGN 1

typedef struct patchEntryRec *patchEntry;
struct patchEntryRec {
    String sym;
    int offset;
    int length;
};

typedef gpuDebugImage lwDebugImage;
typedef patchEntry    lwPatchEntry;

struct lwDwarfInfoRec {
    int is_64bit;
    stdMap_t    stackoffset;                //entryname::varname --> stackoffset
    stdMap_t    dwarfFiles;                 //from ptxParsingState
    String fileTableEntry;                  //file table entry in lw_debug_line_sass --> ".lw_debug_ptx_txt.<checksum>" (only used for SC)
};

lwDwarfInfo lwInitDwarf(int is_64bit)
{
    lwDwarfInfo dwarfInfo;
    stdNEW(dwarfInfo);
    dwarfInfo->is_64bit = is_64bit;
    dwarfInfo->fileTableEntry = NULL;
    return dwarfInfo;
}

// OPTIX_HAND_EDIT
#if 0
/* Returns True, iff symName is a debug section name.
   Else returns False.
*/
static Bool IsDebugSectionName(String symName)
{
    // Each debug_section starts with either ".lw_debug_" or ".debug_".
    // So check if symName starts with one of above strings.
    return !(strncmp(symName, ".lw_debug_", 10) &&
             strncmp(symName, ".debug_", 7));
}

static void renamePtxSection(stdList_t ptxInputs, Bool compileOnly, lwDwarfInfo dwarfInfo)
{
    String fileTableEntry = NULL;
    if (compileOnly) {
        /* For SC, we rename the lw_debug_ptx_txt section by appending the checksum of ptx input,
         * i.e. ".lw_debug_ptx_txt.<checksum>".
         * This unique section will be referred by lw_debug_line_sass section via file table entry. */
        lwDebugImage inputPtxDebugImage = (lwDebugImage)ptxInputs->head;
        unsigned long adler = adler32(0L, NULL, 0);   //Initialize checksum

        stdASSERT((listSize(ptxInputs) == 1) ,("Invalid number of input ptx files"));

        /* Callwlate the checksum for ptx input */
        adler = adler32(adler, inputPtxDebugImage->image, inputPtxDebugImage->size);
        fileTableEntry = (String) stdMALLOC(50);
        sprintf(fileTableEntry, "%s.%lu", DEBUG_PTX_TXT_SECNAME, adler);
    }
    dwarfInfo->fileTableEntry = fileTableEntry;
}

// Following routines are used for debugging purposes and are guarded under CT_DEBUG_DO.
static void printLabelInMap(String s, stdMap_t m)
{
    stdList_t l = mapApply(m, s);
    CT_DEBUG_MSG("dwarf", 5, "label %s\n",s); 
    for (;l; l= l->tail) {
        ptxDwarfLiveRangeMapListNode v = (ptxDwarfLiveRangeMapListNode)l->head;   
        CT_DEBUG_MSG("dwarf", 5, "var %s\n", v->ptxRegisterName);
    }
}

static void printFunLabelInMap(String s, stdMap_t map)
{
    stdMap_t m = mapApply(map, s);
    CT_DEBUG_MSG("dwarf", 5, "Function %s\n", s);
    mapDomainTraverse(m, (stdEltFun)printLabelInMap, m); 
}
    
static void ptxDwarfPrintLiveRangeMap(stdMap_t map)
{
   mapDomainTraverse(map, (stdEltFun)printFunLabelInMap, map); 
}

static void insertEntryInLiveRangeInfoMap(stdMap_t map, String functionName, String labelName, ptxDwarfLiveRangeMapListNode v)
{   
    stdMap_t labelToListMap; 
    stdList_t l = NULL;

    /* Early exit
    *  If either functionName or labelName
    *  then don't insert entry in map so that 
    *  later part of the code won't fail.
    *  Note, by not inserting an entry means we're skipping
    *  live range for this symbol.
    */

    if (!functionName || !labelName) {
        CT_DEBUG_MSG("dwarf", 1, "->WARNING : Not inserting entry in map since info seems incorrect\n");
        return;
    }

    if ((labelToListMap = mapApply(map, functionName))) {
        l = mapApply(labelToListMap, labelName);
    } else {
        labelToListMap = mapNEW(String, 8192);
        CT_DEBUG_MSG("dwarf", 1, "Inserting function %s in live range map\n", functionName);  
        mapDefine(map, functionName, labelToListMap);
    }
    listAddTo(v, &l);
    mapDefine(labelToListMap, labelName, l);
}
// OPTIX_HAND_EDIT
#endif

/*
* Function          : Decode value of packed dwarf data into actual dwarfdata
*                     and information about size of dwarf data and presence
*                     of label.
* Parameters        :  imagePackedBytesptr : encoded value
*                      extractedDwarfData  : Pointer to which extracted
*                                            dwarf data should be placed,
*                      labelAndSizeVal     : Pointer to which value of label and
*                                            size should be placed.
*                      longConstantArr     : Array of 64 bit long constants
* Function Results  : True/False as per success of decoding.
* Comments          : Please refer to the corresponding packing function
                      to know how things are packed.
*/

static Bool ptxDwarfUnpackDwarfData(uInt64 *imagePackedBytesptr,
                                    uInt64 *extractedDwarfData,
                                    uInt64 *labelAndSizeVal,
                                    uInt64 *longConstantArr)
{
    uInt bytesReq, metadata     = 0;
    uInt64 mask                 = 1;
    uInt64 packedDwarfData      = *imagePackedBytesptr;
    // metadata is present at first byte of packed record
    metadata                    = packedDwarfData & 0xff;
    if (metadata) {
        bytesReq                = metadata > DWARF_LABEL_INDICATOR ? sizeof(uInt32) : metadata;
        if (metadata == 8) {
            // when metadata is of size 8 byte the actual data is present in
            // longCnstantArr and index of that array is present in packedDwarfData
            // size of index is equal to 32 bits which is fixed under packing scheme
            bytesReq = sizeof(uInt32);
        }
        *labelAndSizeVal        = metadata;
        packedDwarfData         = packedDwarfData >> 8;
        *extractedDwarfData     = packedDwarfData & ((mask << bytesReq * 8) - 1);
        if (*labelAndSizeVal == 8) {
            // In case of constant of 8 byte, decoded data will represent
            // index in longConstantArr at which actual data is present
            // This index is stored as 32 bit integer in packed record
            *extractedDwarfData = longConstantArr[*extractedDwarfData];
            bytesReq = sizeof(uInt32);
        }
        packedDwarfData         = packedDwarfData >> (8 * bytesReq);
        *imagePackedBytesptr    = packedDwarfData;
        return True;
    }
    // Zero value of metadata indicates end of dwarf records in array element
    return False;
}

// OPTIX_HAND_EDIT static 
char* ptxDwarfCreateByteStream(ptxDwarfSection section, stdVector_t stringVect, stdMap_t dwarfLabelMap)
{
    uInt offset = 0, numberOfVectorElements, i;
    uInt64 imageBytes, labelAndSizeVal, imagePackedBytes, count;
    String imageLabel;
    char *buf, *offset_ptr;
    dwarfLinesRec imageVector;
    String dwarfLabel;

    imageVector = section->dwarfLines;
    numberOfVectorElements = imageVector.dwarfLineBytesTop;
    stdNEW_N(buf, section->size);
    // Extract values of dwarf data and labelAndSize from the encoded values
    // present in dwarfLineBytes
    for (i = 0; i < numberOfVectorElements; ++i) {
        imagePackedBytes = (uInt64) imageVector.dwarfLineBytes[i];
        while (1) {
            Bool hasData = ptxDwarfUnpackDwarfData(&imagePackedBytes, &imageBytes,
                                                   &labelAndSizeVal,
                                                   imageVector.longConstantArr);
            if (!hasData) {
                break;
            }

            stdASSERT(offset < section->size, ("Offset exceeding section size\n"));
            if (labelAndSizeVal >= DWARF_LABEL_INDICATOR) {
                // When label is present imageByte will indicate index of string vector
                labelAndSizeVal = labelAndSizeVal - DWARF_LABEL_INDICATOR;
                imageLabel      = (String)vectorIndex(section->labelVector, imageBytes);
                dwarfLabel      = mapApply(dwarfLabelMap, imageLabel);
                if (dwarfLabel != NULL) {
                    // Colwert DWARF label into "<section_name> + offset" form
                    imageLabel = dwarfLabel;
                }
                // When passing abbrev_offset pass actual offset instead of label
                if (section->sectionType == DEBUG_INFO_SECTION &&
                    (offset_ptr = (char *)stdStringIsPrefix(DEBUG_ABBREV_SECNAME, imageLabel))) {
                    int abbrev_offset;
                    if (*offset_ptr == '+') {
                        sscanf (offset_ptr+1, "%u", &abbrev_offset);
                    } else {
                        abbrev_offset = 0;
                    }
                    stdMEMCOPY_N((Byte *)(buf + offset), &abbrev_offset, labelAndSizeVal);
                } else {
                    count = vectorSize(stringVect);
                    vectorPush(stringVect, imageLabel);
                    CT_DEBUG_MSG("dwarf", 1, "%lld -> %s\n", count, (String)vectorTop(stringVect));
                    stdMEMCOPY_N((Byte *)(buf + offset), &count, labelAndSizeVal);
                }
            } else {
                // Copy number of bytes equal to size of image only
                stdMEMCOPY_N((Byte*)(buf + offset), &imageBytes, labelAndSizeVal);
            }
            offset += labelAndSizeVal;
        }
    }
    return buf;
}

// OPTIX_HAND_EDIT
#if 0
static void ptxDwarfDecodeDebugLocSection(char* buf, uInt sectionSize, int ptrSize, stdVector_t stringVect, 
                                          stdMap_t offsetToFunctionName, stdMap_t ptxDwarfLiveRangeInfoMap)
{
    uInt entryStart, offset = 0;
    unsigned short sizeOfEntry = 0;
    unsigned char opcode = 0; 
    String startLabel   = NULL, endLabel   = NULL;
    String functionName = NULL, ptxRegName = NULL;
    signed long long ptxRegOffset = 0;
    ptxDwarfLiveRangeMapListNode nodeForStartLabel, nodeForEndLabel;
    int slen, regNameLength = 0, regOffsetLength = 0;
    Bool isFirstRecord = True, isLastRecord = False;      // Using for code readability
   
    /* As per DWARF4 Specification,
    * "A location list entry consists of two relative addresses
    * followed by a 2-byte length, followed by a block of contiguous bytes. 
    * The length specifies the number of bytes in the block that follows"
    * so typical format is 
    *     .b<32/64> startLabel
    *     .b<32/64> endLabel
    *     2-byte size
    *     1 byte opcode
    *     ... 
    */
 
    while (offset < sectionSize) { 
        entryStart = offset;
        switch(ptrSize) {
        case 4:
            startLabel = vectorIndex(stringVect, *(uInt*)(buf + offset));
            offset    += ptrSize;    
            endLabel   = vectorIndex(stringVect, *(uInt*)(buf + offset));
            break;
        case 8:
            startLabel = vectorIndex(stringVect, *(uInt64*)(buf + offset));
            offset    += ptrSize;    
            endLabel   = vectorIndex(stringVect, *(uInt64*)(buf + offset));
            break;
        default:
            stdASSERT(0, ("Invalid pointer size"));
        }

        CT_DEBUG_MSG("dwarf", 5, "[loc] Extracted startLabel %s and endLabel %s\n", startLabel, endLabel);
 
        offset += ptrSize;    

        // offset now points to 2 byte-size of block

        sizeOfEntry = *(unsigned short*)(buf + offset);
        stdASSERT(sizeOfEntry > 0, ("Invalid size of entry"));

        offset += 2;

        // offset now points to first byte of the block 
        // Inspect opcode.
        // Lwrrently we are supporting DW_OP_regx, DW_OP_bregx opcodes

        opcode = *(buf + offset);
        stdNEW_N(ptxRegName, REGISTER_NAME_SIZE);
        switch (opcode) {
        case DW_OP_regx:
            offset += 1;
            // offset now points to data block which can be decoded using LEB 128 decoder.
            _dwarf_decode_leb128_nm_long (buf + offset, ptxRegName, REGISTER_NAME_SIZE, &slen); 
            regNameLength = slen;
            regOffsetLength = 0;
            break;
        case DW_OP_bregx:
            offset += 1;
            // offset now points to data block containing 2 LEB128 encoded values of for register name
            // string and register offset.
            _dwarf_decode_leb128_nm_long (buf + offset, ptxRegName, REGISTER_NAME_SIZE, &slen); 
            regNameLength = slen;
            ptxRegOffset = decodeSignedLEB128(buf + offset + regNameLength, &slen);
            regOffsetLength = slen;
            break;
        default:
            CT_DEBUG_MSG("dwarf", 1, "->WARNING : Encountered an unhandled opcode while \
                                       parsing .debug_loc so skipping further DWARF info");
            stdASSERT(0, ("Unhandled opcode\n"));
            return;
        }


        
        stdASSERT(regNameLength > 0, ("ptxRegister name not decoded properly\n"));
        stdASSERT(regNameLength + regOffsetLength  <= sizeOfEntry + 1, ("Encoded size is bigger than LEB128 data"));
        stdASSERT(regNameLength < REGISTER_NAME_SIZE, ("Cannot accommodate register name in buffer\n"));

        offset += sizeOfEntry - 1;             // -1 due to opcode.
        CT_DEBUG_MSG("dwarf", 5, "[loc] ptxRegister name decoded [%s + %lld]\n", ptxRegName, ptxRegOffset);

        stdNEW(nodeForEndLabel);

        if (!isFirstRecord) {

            // DO NOT insert startLabel here because as per the 
            // contract with OCG, we do not need to add symbols to 
            // this start label's list So, insert for end label

            nodeForEndLabel->isFirstDefinition = False;
            nodeForEndLabel->ptxRegisterName   = ptxRegName;
            insertEntryInLiveRangeInfoMap(ptxDwarfLiveRangeInfoMap, functionName, endLabel, nodeForEndLabel);
            CT_DEBUG_MSG("dwarf", 1, "[loc] inserted %s to %s's map in big table\n", endLabel, functionName);

        } else if (isFirstRecord && (functionName = mapApply(offsetToFunctionName, (Pointer)(Address)entryStart))) {

            // 1. Insert for start label
            // For start label attribute, set isFirstDefinition = True so that we insert in
            // declaredList of OCG (through AdddIDagSymbolToDeclaredSymbols)

            isFirstRecord = False;
            stdNEW(nodeForStartLabel); 
            nodeForStartLabel->isFirstDefinition = True;
            nodeForStartLabel->ptxRegisterName   = ptxRegName;
            nodeForStartLabel->isLocList         = True;
            insertEntryInLiveRangeInfoMap(ptxDwarfLiveRangeInfoMap, functionName, startLabel, nodeForStartLabel);
            CT_DEBUG_MSG("dwarf", 1, "[loc] inserted %s to %s's map in live range map\n", startLabel, functionName);

            // 2. Insert for end label
            // For start label attribute, set isFirstDefinition = False so that we insert in
            // extendedSymbolsList of OCG (through AdddIDagSymbolToDeclaredSymbols)

            nodeForEndLabel->isFirstDefinition = False;
            nodeForEndLabel->ptxRegisterName   = ptxRegName;
            nodeForEndLabel->isLocList         = True;
            insertEntryInLiveRangeInfoMap(ptxDwarfLiveRangeInfoMap, functionName, endLabel, nodeForEndLabel);
            CT_DEBUG_MSG("dwarf", 1, "[loc] inserted %s to %s's map in live range map\n", endLabel, functionName);
        } 

        // Two zeros indicate end of live 
        // range so clear temp data and get ready for next loc entry.
        switch(ptrSize) {
        case 4:
            if (*(uInt*)(buf + offset) == 0 && *(uInt*)(buf + offset + ptrSize) == 0)
                isLastRecord = True;
            break;
        case 8:
            if (*(uInt64*)(buf + offset) == 0 && *(uInt64*)(buf + offset + ptrSize) == 0)
                isLastRecord = True;
            break;
        default:
            stdASSERT(False, ("Unexpected size"));
        }

        if (isLastRecord) {
            offset       += ptrSize * 2;
            functionName  = NULL;
            isFirstRecord = True;
            isLastRecord  = False;
            continue;
        }
    }
}

static void ptxDwarfPopulateLiveRangeInfoMap(stdMap_t map, stdMap_t offsetToFunctionName, stdList_t l, stdVector_t stringVect)
{
    ptxDwarfSymbolInfo v;
    ptxDwarfLiveRangeMapListNode nodeForStartLabel, nodeForEndLabel;
    char *c;

    if (!l) {
        CT_DEBUG_MSG("dwarf", 1, "Found empty node, skipping live range info\n");
        return;
    }

    for(; l; l = l->tail) {
        v = (ptxDwarfSymbolInfo)l->head;
        if (!v) {
            CT_DEBUG_MSG("dwarf", 1, "Failed to extract ptxDwarfSymbolInfo, continuing\n");
            stdASSERT(0, ("Empty ptxDwarfSymbolInfo\n"));
            continue;
        }

        if (v->isLocList == 0) {
            stdNEW(nodeForStartLabel);
            stdNEW(nodeForEndLabel);
            
            // For low_pc attribute (of closest DIE ancestor of type
            // DW_TAG_lexical_scope), set isFirstDefinition = True so that we insert in
            // declaredList of OCG (through AdddIDagSymbolToDeclaredSymbols)
            
            nodeForStartLabel->ptxRegisterName   = v->cases.expr.ptxRegisterName;
            nodeForStartLabel->isFirstDefinition = True;
            nodeForStartLabel->isLocList         = 0;
            insertEntryInLiveRangeInfoMap(map, v->functionName, vectorIndex(stringVect, v->cases.expr.startLabelIndex), nodeForStartLabel);

            // For high_pc attribute (of closest DIE ancestor of type
            // DW_TAG_lexical_scope), set isFirstDefinition = False so that we insert in
            // extnodeForEndLabeledSymbolsList of OCG (through AddIDagSymbolToLiveRangeExtnodeForEndLabeledSymbols)

            nodeForEndLabel->ptxRegisterName   = v->cases.expr.ptxRegisterName;
            nodeForEndLabel->isFirstDefinition = False;
            nodeForEndLabel->isLocList         = 0;
            insertEntryInLiveRangeInfoMap(map, v->functionName,  vectorIndex(stringVect, v->cases.expr.endLabelIndex), nodeForEndLabel);

        } else if (v->isLocList == 1) {

            // This means we don't have enough info about
            // this variable and the info will 
            // be available after parsing .debug_loc section
            // so, fill the information in offsetToFunctionName which 
            // will be used by .debug_loc parser to populate 
            // map in ptxParsingState

            if ((c = strchr(vectorIndex(stringVect, v->cases.llist.offsetIntoDebugLocSection), '+'))) {
                mapDefine(offsetToFunctionName, (Pointer)(Address)atoi(c + 1), v->functionName);
                CT_DEBUG_MSG("dwarf", 1, "inserted (%s,%d) into offsetToFunctionName\n", v->functionName, atoi(c+1));
            } else {
                mapDefine(offsetToFunctionName, 0, v->functionName);
                CT_DEBUG_MSG("dwarf", 1, "inserted (%s,%d) into offsetToFunctionName\n", v->functionName, 0);
            }

        }  else {
            stdASSERT(0, ("Unexpected value for locList\n"));    
        }
    }
}

#define STRING_TABLE_SIZE 1000

void ptxDwarfExtractLiveRanges(ptxParsingState p, lwDwarfInfo dwarfInfo)
{
    char *buf;
    stdVector_t stringVect;
    stdList_t list = NULL;
    stdMap_t offsetToFunctionName;
    dwarfStateInfo state;

    ptxDwarfSection abbrevSection, infoSection, locSection; 

    // 1. Decode .debug_abbrev section
    // .debug_abbrev section must be present in debug PTX files.
    abbrevSection = ptxDwarfGetSectionPointer(p, DEBUG_ABBREV_SECTION);

    // 2. Decode .debug_info section.
    // .debug_info section must be present in debug PTX files.
    infoSection = ptxDwarfGetSectionPointer(p, DEBUG_INFO_SECTION);
    
    if (!abbrevSection || !infoSection) {
        return;
    }

    stringVect = vectorCreate(STRING_TABLE_SIZE);
    offsetToFunctionName = mapNEW(uInt, 1000);
    
    // Start inserting from second location in vector 
    // because we want index to start from 1.
    // This is because patching "0" in byte stream will
    // cause decodeLEB128 to fail since it stops  
    // decoding as soon as it sees "0" as byte.
    vectorPush(stringVect, "NULL");

    state = initializeDwarfStateInfo();
    buf = ptxDwarfCreateByteStream(abbrevSection, stringVect, p->internalDwarfLabel);
    decodeDebugAbbrevTable(state, buf, abbrevSection->size, 0);                        // Disabled printing via third arg. 
    stdFREE(buf);

    buf = ptxDwarfCreateByteStream(infoSection, stringVect, p->internalDwarfLabel);
    // Following function populates "list"
    // Disabled priting via last argument.
    decodeDebugInfo(state, buf, infoSection->size, NULL, NULL, DEBUG_INFO_SECNAME, &list, True, False);
    stdFREE(buf);

    // Populate information gathered from debug_info into map.
    ptxDwarfPopulateLiveRangeInfoMap(p->dwarfLiveRangeMap, offsetToFunctionName, list, stringVect);

    // 3. Decode .debug_loc section.
    // .debug_loc may or may not be present in debug PTX files.
    locSection = ptxDwarfGetSectionPointer(p, DEBUG_LOC_SECTION);
    if (locSection) {
        buf = ptxDwarfCreateByteStream(locSection, stringVect, p->internalDwarfLabel);
        // Decode .debug_loc section and populate information in dwarfLiveRangeMap
        ptxDwarfDecodeDebugLocSection(buf, locSection->size, dwarfInfo->is_64bit ? 8 : 4, stringVect, offsetToFunctionName, p->dwarfLiveRangeMap);
        stdFREE(buf);
    } else {
        CT_DEBUG_MSG("dwarf", 1, ".debug_loc not present in PTX\n");
    }

    // Free temporary data structures.
    vectorDelete(stringVect);
    listDelete(list);
    mapDelete(offsetToFunctionName);
    deleteDwarfStateInfo(&state);
    
    CT_DEBUG_DO("dwarf", 5, ptxDwarfPrintLiveRangeMap(p->dwarfLiveRangeMap););
}

/* Look for the first oclwrance of either '+' or '-'
 * Return the pointer to its location. Return NULL if not found.
 */
String findPlusOrMinus(String symName)
{
    while (*symName) {
        switch (*symName) {
        case '-':
        case '+':
            return symName;
        }
        ++symName;
    }
    return NULL;
}

/* Process the diff and return it.
 * Assumes 'symName' to be a destructible copy, don't pass a string which needs to be preserved.
 */
static uInt64 processLabelDiffRecord(ptxParsingState state, String symName, String minusPtr)
{
    String offset1, offset2, sectionPlusOffset1, sectionPlusOffset2;
    uInt64 off1, off2;

    // This splits "Label1-Label2" into 2 strings symName = "Label1"  minusPtr+1 = "Label2"
    *minusPtr = '\0';

    /*
        For instance let label1 -> ".debug_pubnames+39" and label2 -> ".debug_pubnames+4"
        extract the difference in section offsets as 35 (39-4).
    */
    sectionPlusOffset1 = (char *)mapApply(state->internalDwarfLabel, symName);
    sectionPlusOffset2 = (char *)mapApply(state->internalDwarfLabel, minusPtr+1);
    if (!sectionPlusOffset1 || !sectionPlusOffset2) {
        stdCHECK(sectionPlusOffset1, (ptxasDwarfDebugIlwalidLbl, symName));
        stdCHECK(sectionPlusOffset2, (ptxasDwarfDebugIlwalidLbl, minusPtr+1));
        return 0;
    }
    offset1 = strchr(sectionPlusOffset1, '+');
    offset2 = strchr(sectionPlusOffset2, '+');
    if (!offset1 || !offset2) {
        stdASSERT(FALSE, ("Expecting .debug_<section-name>+offset as mapped value of labels"));
        return 0;
    }
    if (!stdEQSTRINGN(sectionPlusOffset1, sectionPlusOffset2, offset1-sectionPlusOffset1)) {
        stdCHECK(FALSE,(ptxasDwarfDebugIllegalLblDifExpr, symName, minusPtr+1));
        return 0;
    }
    sscanf (offset1, "%llu", &off1);
    sscanf (offset2, "%llu", &off2);
    return off1-off2;
}

/* Generate debug info sections => .debug_info*/
static void lwElfGenDebugInfoSections(dwarfStateInfo dwarfState, elfw_t lwelf, DebugInfo *debugHandle,
                                      ptxParsingState state, ptxDwarfSection debugInfoDwarfSec,
                                      Bool isWholeProgComp, lwDwarfInfo dwarfInfo)
{
    stdList_t patches = NULL;   //the list of patches for stack vars
    Int32 size = 0, secOffset = 0, i;
    uInt64 labelAndSizeVal, imageBytes, imagePackedBytes;
    String image = NULL, ptximage = NULL;
    elfWord secidx = 0;
    stdList_t entry_range = NULL;
    String dwarfLabel;

    if (!dwarfState->num_abbrev_entries) {
        // .debug_abbrev section missing. Skipping decode of .debug_info section
        reportSkipDebugInfoDecode();
        return;
    }

    secidx = elfw_add_data_section(lwelf, (String)DEBUG_INFO_SECNAME, image, 1, size);
    size = debugInfoDwarfSec->size;

    stdNEW_N(image, size);
    if (isWholeProgComp)
        stdNEW_N(ptximage, size);

    for (i = 0; i < debugInfoDwarfSec->dwarfLines.dwarfLineBytesTop; ++i) {
        imagePackedBytes = (uInt64) debugInfoDwarfSec->dwarfLines.dwarfLineBytes[i];
        while (1) {
            Bool hasData = ptxDwarfUnpackDwarfData(&imagePackedBytes, &imageBytes,
                                                   &labelAndSizeVal,
                                                   debugInfoDwarfSec->dwarfLines.longConstantArr);
            if (!hasData) {
                break;
            }

            if (labelAndSizeVal >= DWARF_LABEL_INDICATOR) {
                labelAndSizeVal     = labelAndSizeVal - DWARF_LABEL_INDICATOR;
                String  symName     = stdCOPYSTRING(vectorIndex(debugInfoDwarfSec->labelVector,
                                                     imageBytes));
                uInt    len         = labelAndSizeVal;
                uInt64  b8value     = 0;
                uInt32  b4value     = 0;
                elfWord reltype     = 0;
                uInt64  labelOffset = 0;
                Bool IsLabelPlusOffset = False;
                String opPtr = NULL;
                ptxSymLocInfo syminfo;
                dwarfLabel      = mapApply(state->internalDwarfLabel, symName);
                if (dwarfLabel != NULL) {
                    // Colwert DWARF label into "<section_name> + offset" form
                    symName = dwarfLabel;
                }

                if (len == 4) {
                    reltype = elfw_is_merlwry(lwelf) ? R_MERLWRY_ABS32 : R_LWDA_32;
                } else if (len == 8) {
                    reltype = elfw_is_merlwry(lwelf) ? R_MERLWRY_ABS64 : R_LWDA_64;
                } else {
                    stdASSERT(0, ("Invalid label type in debug_info section"));
                }

                /* Check if dwarf is of the form Label+offset or Label1-Label2 */
                opPtr = findPlusOrMinus(symName);

                /* Handle Label+offset syntax in Dwarf data */
                if (opPtr && *opPtr == '+') {
                    *opPtr = '\0';
                    opPtr++;
                    sscanf(opPtr, "%llu", &labelOffset);
                    dwarfLabel = mapApply(state->internalDwarfLabel, symName);
                    if (dwarfLabel != NULL) {
                        uInt64 dwarfLabelOffset;
                        // DWARF label+offset is present
                        opPtr = strchr(dwarfLabel, '+');
                        *opPtr = '\0';
                        opPtr++;
                        sscanf(opPtr, "%llu", &dwarfLabelOffset);
                        symName = dwarfLabel;
                        labelOffset += dwarfLabelOffset;
                    }
                    IsLabelPlusOffset = True;
                }

                /* Handle Label1-Label2 syntax in Dwarf data. Retrieve the .<section-name>+offset
                 * value for each label and output the difference in offsets.
                 */
                if (opPtr && *opPtr == '-') {
                    imageBytes = processLabelDiffRecord(state, symName, opPtr);
                    stdMEMCOPY_N((Byte *)(image + secOffset), &imageBytes, len);
                    if (isWholeProgComp) {
                        stdMEMCOPY_N((Byte *)(ptximage + secOffset), &imageBytes, len);
                    }
                } else if ((syminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->sassAddresses, symName)) != NULL) {
                    /* Output the value if we know symbol's address */
                    ptxSymLocInfo ptxsyminfo = NULL;

                    if (isWholeProgComp)
                        ptxsyminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->locationLabels, symName);

                    if(!syminfo->isParam) {
                        elfWord entrysym = 0;
                        elfByte sttType;
                        entrysym = elfw_lookup_symbol_index(lwelf, syminfo->entry);
                        stdASSERT(entrysym, ("size info in debug sec is inaclwrate"));

                        elfw_get_symbol_attributes(lwelf, entrysym, &sttType, NULL, NULL);
                        if (elfw_is_merlwry(lwelf) && sttType == STT_FUNC) {
                            /* On Merlwry for function relocations on debug_info, generate *ABS_PROG_REL* relocs.
                             * This is needed since relocation may be representing address of label inside function.
                             * Such relocation needs to be colwerted to ABS_PROG_REL so that addend gets adjusted to
                             * corresponding SASS offset.
                             */
                            reltype = (reltype == R_MERLWRY_ABS32) ? R_MERLWRY_ABS_PROG_REL32 : R_MERLWRY_ABS_PROG_REL64;
                        }
                        elfw_add_reloca(lwelf, reltype, entrysym, secidx, secOffset, syminfo->offset);
                    }

                    switch (len) {
                    case 4:
                        b4value = (uInt32)(syminfo->offset + labelOffset);
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b4value, len);
                        if (isWholeProgComp) {
                            b4value = (uInt32)(ptxsyminfo->offset + labelOffset);
                            stdMEMCOPY_N((Byte *)(ptximage + secOffset), &b4value, len);
                        }
                        break;
                    case 8:
                        b8value = (uInt64)(syminfo->offset + labelOffset);
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b8value, len);
                        if (isWholeProgComp) {
                            b8value = (uInt64)(ptxsyminfo->offset + labelOffset);
                            stdMEMCOPY_N((Byte *)(ptximage + secOffset), &b8value, len);
                        }
                        break;
                    default:
                        stdASSERT(False, ("Unexpected size"));
                    } 
                }
                /* For parameters in CBANK, we need to generate the relocator and patch the offset
                 * TODO: generate relocator with addend */
                else if((syminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->paramOffset, symName)) != NULL) {
                    switch (len) {
                    case 4:
                        b4value = (uInt32)(syminfo->offset + labelOffset);
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b4value, len);
                        if (isWholeProgComp) {
                            stdMEMCOPY_N((Byte *)(ptximage + secOffset), &b4value, len);
                        }
                        break;
                    case 8:
                        b8value = (uInt64)(syminfo->offset + labelOffset);
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b8value, len);
                        if (isWholeProgComp) {
                            stdMEMCOPY_N((Byte *)(ptximage + secOffset), &b8value, len);
                        }
                        break;
                    default:
                        stdASSERT(False, ("Unexpected size"));
                    }
                }
                /* Generate patch info entry if we find it *could* be a stack variable */
                else if(mapIsDefined(dwarfInfo->stackoffset, symName)) {
                    lwPatchEntry patch;

                    stdASSERT(!IsLabelPlusOffset, ("Invalid dwarf value of form '<stack variable>+offset'"));

                    stdNEW(patch);

                    patch->sym = stdCOPYSTRING(symName);
                    patch->offset = secOffset;
                    patch->length = len;

                    listAddTo(patch, &patches);
                }
                /* Generate relocations if we are not sure about the sass address */
                else {
                    elfWord varsym = elfw_lookup_symbol_index(lwelf, symName);
                    VariableInfo* VI;
                    if (varsym == NULL_ELFW_INDEX) {
                        if (IsDebugSectionName(symName) && strcmp(symName, DEBUG_LINE_SECNAME) != 0) {
                            /* special case finding reference to a debug section which is defined later,
                             * except DEBUG_LINE_SECNAME which should have already been defined */
                            varsym = elfw_add_data_section(lwelf, symName, NULL, 1, 0);
                        } else {
                            varsym = elfw_add_symbol(lwelf, symName, STT_NOTYPE, STB_LOCAL, 0, 0, 0, 1, 0);
                        }
                    }
                    VI = lookupVariableInfo(debugHandle->SymHandle, symName);
                    if (VI && VI->Space == SPACE_Constant) {
                        /* For constant symbols emit R_LWDA_G* relocation since debugger
                         * patches generic address of constant symbols not specific address 
                         * in debug information */
                        if (elfw_is_merlwry(lwelf)) {
                            // For now Merlwry supports only 64bit relocs.
                            // FIXME: Handle 32bit generic addresses by adding R_MERLWRY_G32
                            reltype = R_MERLWRY_G64;
                        } else {
                            reltype = (reltype == R_LWDA_32) ? R_LWDA_G32 : R_LWDA_G64;
                        }
                    }

                    /* Generate rela so linker can put section offset into addend */
                    elfw_add_reloca(lwelf, reltype, varsym, secidx, secOffset, labelOffset);
                }
                stdFREE(symName);
            } else {
                stdMEMCOPY_N((Byte *)(image + secOffset), &imageBytes, labelAndSizeVal);
                if (isWholeProgComp) {
                    stdMEMCOPY_N((Byte *)(ptximage + secOffset), &imageBytes, labelAndSizeVal);
                }
            }

            secOffset += labelAndSizeVal;
            stdASSERT(secOffset <= size, ("Invalid debug_info section size"));
        }
    }

    stdASSERT(secOffset == size, ("Problem in adding imageList to image"));

    /* we just decode .debug_info sec because its patterns are the same as .lw_debug_info_ptx */
    if (patches)
        entry_range = elfLightWeightDecoder(dwarfState, image, image + size, dwarfInfo->is_64bit ? sizeof(unsigned long long) : sizeof(unsigned int));

    while (patches) {
        stdList_t tail = patches->tail;
        lwPatchEntry lwrpatch = (lwPatchEntry)patches->head;
        entry_range_debug_info_ptr lwrentry = NULL;
        LocalVarOffset *lwrvarinfo = (LocalVarOffset*)mapApply(dwarfInfo->stackoffset, lwrpatch->sym);

        if(entry_range)
            lwrentry = (entry_range_debug_info_ptr)entry_range->head;

        /* both the patches and the entries are sorted */
        while(lwrentry && lwrentry->entry_start > lwrpatch->offset)
        {
            entry_range = entry_range->tail;
            if (!entry_range) {
                stdASSERT(False, ("Unexpected end of subprogram list"));
                lwrentry = NULL;
                break;
            }
            lwrentry = (entry_range_debug_info_ptr)entry_range->head;
        }

        if (!lwrentry) {
            stdASSERT(False, ("Valid subprogram entry must be present for every patch"));
            break;
        }
        if(strcmp(lwrentry->entry_name, lwrpatch->sym) &&
          (*(char*)(image + lwrpatch->offset - 1) == DW_OP_addr))
        {
            /* The patch location is originally for the value of the DW_OP_addr, 
               we check the OP here */
            int nbytes; 
            *(char*)(image + lwrpatch->offset - 1) = DW_OP_fbreg;

            if (EncodeSignedLeb128(lwrvarinfo->offset, &nbytes,
                (char *)(image + lwrpatch->offset), 255) == DW_DLV_ERROR)
            {
                stdASSERT(0, ("Error when patching stack variables in .debug_info section"));
            }
            /* If nbytes < lwrpatch->length,  we use DW_OP_nop padding */
            while(nbytes < lwrpatch->length) {
                *(char*)(image + lwrpatch->offset + nbytes) = DW_OP_nop;
                nbytes++;
            }
            if (isWholeProgComp) {
                /* patch both DW_OP_fbreg and the block in the .lw_debug_info_ptx section */
                stdMEMCOPY_N((Byte *)(ptximage + lwrpatch->offset - 1), (Byte *)(image + lwrpatch->offset -1), 
                    lwrpatch->length + 1);
            }
        } else {
            elfWord varsym;
            elfWord reltype = (lwrpatch->length == 4 ? R_LWDA_32 : R_LWDA_64);
            if ((varsym = elfw_lookup_symbol_index(lwelf, lwrpatch->sym)) == NULL_ELFW_INDEX) {
                varsym = elfw_add_symbol(lwelf, lwrpatch->sym, STT_NOTYPE, STB_GLOBAL, 0, 0, 0, 1, 0);
            }
            elfw_add_reloc(lwelf, reltype, varsym, secidx, lwrpatch->offset);
        }
        stdFREE(lwrpatch->sym);
        stdFREE(lwrpatch);
        patches = tail;
    }

    elfw_add_data_to_section(lwelf, secidx, NULL_ELFW_INDEX, image, 0, DEFAULT_ALIGN, size);
}

/* Generate misc debug sections => .debug_abbrev, .debug_pubnames, .debug_loc, .debug_ranges */
static String lwElfGenDebugMiscSections(elfw_t lwelf, ptxParsingState state, ptxDwarfSection debugMiscDwarfSec, lwDwarfInfo dwarfInfo)
{
    stdList_t patches = NULL;   //the list of patches for stack vars
    dwarfLinesRec imageVector;
    Int32 size = 0, secOffset = 0;
    uInt64 labelAndSizeVal = 0, i, noOfElements, imageBytes = 0;
    uInt64 imagePackedBytes = 0;
    String image = NULL;
    elfWord secidx = 0;
    String dwarfLabel;

    if ((secidx = elfw_lookup_symbol_index(lwelf, debugMiscDwarfSec->name)) == NULL_ELFW_INDEX) {
        secidx = elfw_add_data_section(lwelf, debugMiscDwarfSec->name, image, 1, size);
    }

    imageVector = debugMiscDwarfSec->dwarfLines;
    size = debugMiscDwarfSec->size;

    if (imageVector.dwarfLineBytesTop == 0) {
        stdASSERT(size == 0, ("Debug section size should be zero"));
        return NULL;
    }

    stdNEW_N(image, size);
    noOfElements = imageVector.dwarfLineBytesTop;

    for (i = 0; i < noOfElements; ++i) {
        imagePackedBytes = (uInt64) imageVector.dwarfLineBytes[i];

        while (1) {
            Bool hasData = ptxDwarfUnpackDwarfData(&imagePackedBytes, &imageBytes,
                                                   &labelAndSizeVal,
                                                   imageVector.longConstantArr);
            if (!hasData) {
                break;
            }

            if (labelAndSizeVal >= DWARF_LABEL_INDICATOR) {
                labelAndSizeVal = labelAndSizeVal - DWARF_LABEL_INDICATOR;
                String  symName = (String) vectorIndex(debugMiscDwarfSec->labelVector,
                                                       imageBytes);
                String opPtr = NULL;
                uInt    len     = labelAndSizeVal;
                uInt64  b8value = 0;
                uInt32  b4value = 0;
                elfWord reltype = 0;
                ptxSymLocInfo syminfo;
                dwarfLabel      = mapApply(state->internalDwarfLabel, symName);
                if (dwarfLabel != NULL) {
                    // Colwert DWARF label into "<section_name> + offset" form
                    symName = dwarfLabel;
                }

                if (len == 4) {
                    reltype = elfw_is_merlwry(lwelf) ? R_MERLWRY_ABS32 : R_LWDA_32;
                } else if (len == 8) {
                    reltype = elfw_is_merlwry(lwelf) ? R_MERLWRY_ABS64 : R_LWDA_64;
                } else {
                    stdASSERT(0, ("Invalid label type in debug section"));
                }

                /* Detect Label1-Label2 syntax in Dwarf data. */
                opPtr = findPlusOrMinus(symName);

                /* Handle Label1-Label2 syntax in Dwarf data. Retrieve the .<section-name>+offset
                 * value for each label and output the difference in offsets.
                 */
                if (opPtr && *opPtr == '-') {
                    int offset = opPtr - symName;
                    /* Make a destructible copy of symName if necessary */
                    symName = stdCOPYSTRING(symName);
                    /* update opPtr to point to the '-' in the new copy */
                    opPtr = symName+offset;
                    imageBytes = processLabelDiffRecord(state, symName, opPtr);
                    stdMEMCOPY_N((Byte *)(image + secOffset), &imageBytes, len);
                    stdFREE(symName);
                } else if ((syminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->sassAddresses, symName)) != NULL) {
                    /* Output the value if we know symbol's address */
                    elfWord entrysym = elfw_lookup_symbol_index(lwelf , syminfo->entry);
                    elfByte sttType;
                    elfw_get_symbol_attributes(lwelf, entrysym, &sttType, NULL, NULL);
                    if (elfw_is_merlwry(lwelf) && sttType == STT_FUNC) {
                        /* On Merlwry for function relocations on debug_info, generate *ABS_PROG_REL* relocs.
                         * This is needed since relocation may be representing address of label inside function.
                         * Such relocation needs to be colwerted to ABS_PROG_REL so that addend gets adjusted to
                         * corresponding SASS offset.
                         */
                        reltype = (reltype == R_MERLWRY_ABS32) ? R_MERLWRY_ABS_PROG_REL32 : R_MERLWRY_ABS_PROG_REL64;
                    }

                    stdASSERT(entrysym, ("size info in debug sec is inaclwrate"));
                    elfw_add_reloc(lwelf, reltype, entrysym, secidx, secOffset);

                    switch (len) {
                    case 4:
                        b4value = (uInt32)syminfo->offset;
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b4value, len);
                        break;
                    case 8:
                        b8value = (uInt64)syminfo->offset;
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b8value, len);
                        break;
                    default:
                        stdASSERT(False, ("Unexpected size"));
                    }
                }
                /* For parameters in CBANK, we need to generate the relocator and patch the offset
                 * TODO: generate relocator with addend */
                else if((syminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->paramOffset, symName)) != NULL) {
                    switch (len) {
                    case 4:
                        b4value = (uInt32)syminfo->offset;
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b4value, len);
                        break;
                    case 8:
                        b8value = (uInt64)syminfo->offset;
                        stdMEMCOPY_N((Byte *)(image + secOffset), &b8value, len);
                        break;
                    default:
                        stdASSERT(False, ("Unexpected size"));
                    }
                }
                /* Generate patch info entry if we find it *could* be a stack variable */
                else if(mapIsDefined(dwarfInfo->stackoffset, symName)) {
                    lwPatchEntry patch;

                    stdNEW(patch);

                    patch->sym = stdCOPYSTRING(symName);
                    patch->offset = secOffset;
                    patch->length = len;

                    listAddTo(patch, &patches);
                }
                /* Generate relocations if we are not sure about the sass address */
                else {
                    elfWord varsym;
                    if ((varsym = elfw_lookup_symbol_index(lwelf, symName)) == NULL_ELFW_INDEX) {
                        varsym = elfw_add_symbol(lwelf, symName, STT_NOTYPE, STB_LOCAL, 0, 0, 0, 1, 0);
                    }
                    /* use rela so when section is resolved
                     * can put section offset in addend */
                    elfw_add_reloca(lwelf, reltype, varsym, secidx, secOffset, 0);
                }
            } else {
                stdMEMCOPY_N((Byte *)(image + secOffset), &imageBytes, labelAndSizeVal);
            }

            secOffset += labelAndSizeVal;
            stdASSERT(secOffset <= size, ("Invalid debug section size"));
       }
    }

    stdASSERT(secOffset == size, ("Problem in adding imageList to image"));

    while (patches) {
        stdList_t tail = patches->tail;
        lwPatchEntry lwrpatch = (lwPatchEntry)patches->head;
        LocalVarOffset *lwrvarinfo = (LocalVarOffset*)mapApply(dwarfInfo->stackoffset, lwrpatch->sym);

        if((*(char*)(image + lwrpatch->offset - 1) == DW_OP_addr)) {
            /* The patch location is originally for the value of the DW_OP_addr,
             * we check the OP here */
            int nbytes; 
            *(char*)(image + lwrpatch->offset - 1) = DW_OP_fbreg;

            if (EncodeSignedLeb128(lwrvarinfo->offset, &nbytes,
                (char *)(image + lwrpatch->offset), 255) == DW_DLV_ERROR)
            {
                stdASSERT(0, ("Error when patching stack variables in debug section"));
            }
            /* If nbytes < lwrpatch->length,  we use DW_OP_nop padding */
            while(nbytes < lwrpatch->length) {
                *(char*)(image + lwrpatch->offset + nbytes) = DW_OP_nop;
                nbytes++;
            }
        } else {
            elfWord reltype, varsym;
            if ((varsym = elfw_lookup_symbol_index(lwelf, lwrpatch->sym)) == NULL_ELFW_INDEX) {
                varsym = elfw_add_symbol(lwelf, lwrpatch->sym, STT_NOTYPE, STB_GLOBAL, 0, 0, 0, 1, 0);
            }
            reltype = (lwrpatch->length == 4 ? R_LWDA_32 : R_LWDA_64);
            elfw_add_reloc(lwelf, reltype, varsym, secidx, lwrpatch->offset);
        }
        stdFREE(lwrpatch->sym);
        stdFREE(lwrpatch);
        patches = tail;
    }
    elfw_add_data_to_section(lwelf, secidx, NULL_ELFW_INDEX, image, 0, DEFAULT_ALIGN, size);

    return image;
}

/* Generate all the dwarf sections in ELF file */
static void lwElfGenDwarfSections(elfw_t lwelf, DebugInfo *debugHandle,
                                  ptxParsingState state, Bool isWholeProgComp,
                                  lwDwarfInfo dwarfInfo)
{
    stdList_t dwarfSecList = state->dwarfSections;
    dwarfStateInfo dwarfState = initializeDwarfStateInfo();
    /* First build and decode ".debug_abbrev" section, which is used to decode ".debug_info" section */
    ptxDwarfSection abbrevSection = ptxDwarfGetSectionPointer(state, DEBUG_ABBREV_SECTION);
    if (abbrevSection) {
        String debugAbbrevImage = lwElfGenDebugMiscSections(lwelf, state, abbrevSection, dwarfInfo);
        decodeDebugAbbrevTable(dwarfState, debugAbbrevImage, abbrevSection->size, 0);
    }

    /* Generate all other dwarf sections, except already generated ".debug_line and .debug_str" */
    dwarfSecList = state->dwarfSections;
    while (dwarfSecList) {
        ptxDwarfSection lwrPtxDwarfSection = (ptxDwarfSection) dwarfSecList->head;

        if (lwrPtxDwarfSection->sectionType == DEBUG_INFO_SECTION) {
            lwElfGenDebugInfoSections(dwarfState, lwelf, debugHandle, state, lwrPtxDwarfSection, isWholeProgComp, dwarfInfo);
        } else if (lwrPtxDwarfSection->sectionType != DEBUG_ABBREV_SECTION &&
                   lwrPtxDwarfSection->sectionType != DEBUG_LINE_SECTION &&
                   lwrPtxDwarfSection->sectionType != DEBUG_STR_SECTION)
        {
            lwElfGenDebugMiscSections(lwelf, state, lwrPtxDwarfSection, dwarfInfo);
        }
        dwarfSecList = dwarfSecList->tail;
    }
    deleteDwarfStateInfo(&dwarfState);
}

/* Process input ptx string and generate image for ".lw_debug_ptx_txt" section */
static String elfProcessPtxTxt(lwDebugImage inputPtxDebugImage, uInt *finalSize)
{
    String origImage, finalImage;
    uInt origSize, finalImageOff;
    String buf;
    uInt pending = 0;
    uInt index, bufSize;
    Bool isLastLine = False;

    origImage = (String)inputPtxDebugImage->image;
    origSize  = inputPtxDebugImage->size;

    stdNEW_N(finalImage, origSize);
    finalImageOff = 0;
    String savedStr;
    while (1) {
        index = strcspn(origImage, "\n");
        if (origImage[index] != '\n')
            isLastLine = True;
        origImage[index] = '\0';

        buf = stdSTRTOK(origImage + strspn(origImage, " \t"), "\r", &savedStr);
        if (buf == NULL ||
            stdIS_PREFIX("#", buf) ||
            stdIS_PREFIX("//", buf) ||
            (stdIS_PREFIX(".loc", buf) && !stdIS_PREFIX(".local", buf)) ||
            stdIS_PREFIX(".file", buf) ||
            stdIS_PREFIX("@@DWARF", buf) ||
            stdIS_PREFIX(".b8", buf) ||
            stdIS_PREFIX(".b32", buf) ||
            stdIS_PREFIX(".b64", buf))
        {
            pending++;
        } else {
            bufSize = strlen(buf) + 1;
            stdMEMCOPY_N((Byte *)(finalImage + finalImageOff + pending), buf, bufSize);
            finalImageOff += pending + bufSize;
            pending = 0;
        }
        if (isLastLine) break;

        origImage = origImage + index + 1;
    }
    *finalSize = finalImageOff;
    return finalImage;
}

/* Generate ".lw_debug_ptx_txt" section */
static void lwElfGenDebugPtxTextSections(elfw_t lwelf, stdList_t ptxInputs, lwDwarfInfo dwarfInfo)
{
    uInt size = 0;
    String image = NULL;

    stdASSERT((listSize(ptxInputs) == 1) ,("Invalid number of input ptx files"));

    image = elfProcessPtxTxt(ptxInputs->head, &size);

    if (size) {
        String secName = dwarfInfo->fileTableEntry ? dwarfInfo->fileTableEntry : DEBUG_PTX_TXT_SECNAME;
        elfw_add_data_section(lwelf, secName, image, 1, size);
    }
}

void lwObjDwarfCreate(elfw_t lwelf, DebugInfo *debugHandle, lwDwarfInfo dwarfInfo,
                      ptxParsingState state, stdList_t ptxInputs, Bool compileOnly,
                      Bool deviceDebug, Bool lineInfoOnly, Bool forceDebugFrame,
                      Bool suppressDebugInfo)
{
    // copy data from debugHandle structure that is needed for PTX processing
    dwarfInfo->stackoffset = debugHandle->StackOffsets;

    if (forceDebugFrame || deviceDebug) {
        addElfDebugFrame(debugHandle, lwelf);
    }

    if (suppressDebugInfo) return;

    if (lineInfoOnly || deviceDebug) {
        ptxDwarfSection lwrPtxDwarfSection;
        renamePtxSection(ptxInputs, compileOnly, dwarfInfo);
        addElfDebugLine(debugHandle, lwelf, state->dwarfFiles,
                        dwarfInfo->fileTableEntry);
        lwElfGenDebugPtxTextSections(lwelf, ptxInputs, dwarfInfo);
        // Generate .debug_str section when -lineinfo is specified as function_name
        // attribute in .debug_line refers to string present in .debug_str section
        if ((lwrPtxDwarfSection = ptxDwarfGetSectionPointer(state, DEBUG_STR_SECTION))) {
            lwElfGenDebugMiscSections(lwelf, state, lwrPtxDwarfSection, dwarfInfo);
        }
    }
    if (deviceDebug) {
        addElfDebugRegSass(debugHandle, lwelf);
        addElfDebugRegType(debugHandle, lwelf);
        lwElfGenDwarfSections(lwelf, debugHandle, state, !compileOnly, dwarfInfo);
    }
}
// OPTIX_HAND_EDIT
#endif
