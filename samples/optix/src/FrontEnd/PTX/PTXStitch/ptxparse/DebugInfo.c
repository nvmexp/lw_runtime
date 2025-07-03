// LWIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2016-2021, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
// LWIDIA_COPYRIGHT_END

// Definitions of debug information for elf generation

#include "DebugInfo.h"
// OPTIX_HAND_EDIT
#if 0
#include "DebugInfoPrivate.h"
#include "stdLocal.h"
#include "dwarf_interface.h"
#include "ucodeToElfMessageDefs.h"

/************* Initialization and Deletion **/
static void initializeLineState(LineState *LS) {
  stdMEMCLEAR(LS);
  LS->lwrrent_file = 0;
  LS->program_length = 0;
  LS->is_set_file = 0;
  LS->lwrrent_addr = -1;
  LS->lwrrent_col = 0;
  LS->lwrrent_line = 1;
  LS->lwrrent_context = 0;
  LS->lwrrent_func_address = 0;
  LS->debug_line_table.dwarf_version = 2;
  LS->debug_line_table.min_inst_length = 1;
  LS->debug_line_table.is_stmt = 1;
  LS->debug_line_table.line_base = -5;
  LS->debug_line_table.line_range = 14;
  LS->debug_line_table.first_special_opcode = DW_NUM_OPCODES  + 1;
  LS->debug_line_table.opcode_args[DW_LNS_copy] = 0;
  LS->debug_line_table.opcode_args[DW_LNS_advance_pc] = 1;
  LS->debug_line_table.opcode_args[DW_LNS_advance_line] = 1;
  LS->debug_line_table.opcode_args[DW_LNS_set_file] = 1;
  LS->debug_line_table.opcode_args[DW_LNS_set_column] = 1;
  LS->debug_line_table.opcode_args[DW_LNS_negate_stmt] = 0;
  LS->debug_line_table.opcode_args[DW_LNS_set_basic_block] = 0;
  LS->debug_line_table.opcode_args[DW_LNS_const_add_pc] = 0;
  LS->debug_line_table.opcode_args[DW_LNS_fixed_advance_pc] = 1;
  LS->debug_line_table.dir_length = 0;
  LS->debug_line_table.file_length = 0;
  LS->debug_line_table.program_length = 0;
  LS->debug_line_table.max_dir_length = MAX_DIR_SIZE;
  LS->debug_line_table.max_file_length = MAX_FILE_SIZE;
  LS->debug_line_table.max_program_length = MAX_PROGRAM_SIZE;
  LS->debug_line_table.generates_inline_info = False;
  stdNEW_N(LS->debug_line_table.directory_buf, MAX_DIR_SIZE);
  stdNEW_N(LS->debug_line_table.file_buf, MAX_FILE_SIZE);
  stdNEW_N(LS->debug_line_table.statement_program, MAX_PROGRAM_SIZE);
  stdNEW_N(LS->debug_line_table.entry_pos, MAX_ENTRIES);
  LS->debug_line_table.num_entry=0;
  LS->debug_line_table.max_entry=MAX_ENTRIES;
}

extern DebugInfo* beginDebugInfo(SymInfo *SI,
                                 FPTR_GetSourceInfoFromLine GetSourceFromLine)
{
  DebugInfo *DI;
  stdNEW(DI);
  DI->SymHandle = SI;
  SI->IsDebug = True;
  DI->GetSourceInfoFromLine = GetSourceFromLine;
  initializeLineState(&DI->LineStates[SECTION_Line]);
  initializeLineState(&DI->LineStates[SECTION_LineSass]);
  DI->BlockIdOffsets = mapNEW(uInt, 1024);
  DI->SassAddresses = mapNEW(String, 1024);
  DI->LocationLabels = mapNEW(String, 1024);
  DI->StackOffsets = mapNEW(String, 128);
  DI->Frame.Image = DI->RegSass.Image = NULL;
  DI->Frame.Size = DI->RegSass.Size = 0;
  return DI;
}

extern void endDebugInfo(DebugInfo *DI) {
  mapDelete(DI->BlockIdOffsets);
  mapDelete(DI->SassAddresses);
  mapDelete(DI->LocationLabels);
  mapDelete(DI->StackOffsets);
  if (DI->Frame.Image) stdFREE(DI->Frame.Image);
  if (DI->RegSass.Image) stdFREE(DI->RegSass.Image);
  stdFREE(DI->LineStates[SECTION_Line].debug_line_table.directory_buf);
  stdFREE(DI->LineStates[SECTION_Line].debug_line_table.file_buf);
  stdFREE(DI->LineStates[SECTION_Line].debug_line_table.statement_program);
  stdFREE(DI->LineStates[SECTION_Line].debug_line_table.entry_pos);
  stdFREE(DI->LineStates[SECTION_LineSass].debug_line_table.directory_buf);
  stdFREE(DI->LineStates[SECTION_LineSass].debug_line_table.file_buf);
  stdFREE(DI->LineStates[SECTION_LineSass].debug_line_table.statement_program);
  stdFREE(DI->LineStates[SECTION_LineSass].debug_line_table.entry_pos);
  listObliterate(DI->AllocedMemory, Nil);
  stdFREE(DI);
}

/************* Process UCode *****************/

static Bool isDwarfEndLabel(String Name) {
  // TODO: eventually, this should be decided by a database constructed by
  // the client dwarf producer.
  // For now, just do a string compare based on our internal PTX standard.
  return Name && stdIS_PREFIX("$LDWend", Name);
}

static void growEntryBuffer(LineState *LS)
{
    int oldSize, newSize;
    EntryPos * newBuf;

    oldSize = LS->debug_line_table.max_entry * sizeof(EntryPos);
    newSize = 2 * oldSize;
    stdNEW_N(newBuf, newSize);
    memcpy(newBuf, LS->debug_line_table.entry_pos, oldSize);
    stdFREE(LS->debug_line_table.entry_pos);
    LS->debug_line_table.entry_pos = newBuf;
    LS->debug_line_table.max_entry = LS->debug_line_table.max_entry * 2;
}

static void adjustProgramBuf(LineState *LS, int size)
{
   if ((LS->program_length + size) >= 
       (LS->debug_line_table.max_program_length - 1)) {
       char * tmp_buf;
       char * tmp_free;
       unsigned long long tmp_size;

       tmp_size = (LS->debug_line_table.max_program_length) * 2;
       stdNEW_N(tmp_buf, tmp_size);
       memcpy(tmp_buf, LS->debug_line_table.statement_program,
              LS->debug_line_table.max_program_length);
       tmp_free = LS->debug_line_table.statement_program;
       LS->debug_line_table.statement_program = tmp_buf;
       LS->debug_line_table.max_program_length = tmp_size;
       stdFREE(tmp_free);
   }
}

/*------------------------------------------------------------
        Given address advance and line advance, it gives
        either special opcode, or a number < 0

        addr_adv should not be 0, as otherwise, we cannot
        tell the difference from a standard opcode

        addr_adv == 0 line_adv != 0   -->  use advance_line
        addr_Adv != 0 line_adv == 0   -->  use advance_addr
        both non zero, try to use special opcode
------------------------------------------------------------*/
static int _dwarf_compute_special_opcode(Dwarf_Unsigned addr_adv, int line_adv,
                                         uInt opcode_base)
{
    int opc;

    addr_adv = addr_adv / DW_MIN_INST_LENGTH;
    if (line_adv == 0 && addr_adv == 0) {
        return 0;
    }
    if (line_adv >= DW_LINE_BASE && line_adv < DW_LINE_BASE + DW_LINE_RANGE) {
        opc = (int) ((line_adv - DW_LINE_BASE) + (addr_adv * DW_LINE_RANGE) +
                     opcode_base);
        if (opc > 255) {
            return -1;
        }
        return opc;
    } else {
        return -1;
    }
}

static void generateStatementProgram(DebugInfo *DI, LineState *LS, int Count, 
                                     LWuCode_DebugLineTable *Table,
                                     cString Entry, Bool Is64bit,
                                     InlineLocInfo *InlineLocTable) {
    int nbytes, k, progRowId = 0;
    char encode_buf[256];
    int *realContext;
    // real Context represents mapping of row-id in table to the row-id
    // in statement program
    stdNEW_N(realContext, Count + 1);
    // save the function entry into the Table 
    if ((LS->debug_line_table.num_entry + 1) >= LS->debug_line_table.max_entry) {
        growEntryBuffer(LS);
    }

    stdNEW_N(LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].entry, (strlen(Entry)+1));
    listAddTo(LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].entry, &DI->AllocedMemory);
    stdMEMSET_N(LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].entry, 0, strlen(Entry)+1);
    strcpy(LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].entry, Entry);

    adjustProgramBuf(LS, 11); // works for both 32 or 64-bit platform 
    if (Is64bit) {
        LS->debug_line_table.statement_program[LS->program_length++] = 0; // extended opcode
        LS->debug_line_table.statement_program[LS->program_length++] = 9; // following block length
        LS->debug_line_table.statement_program[LS->program_length++] = DW_LNE_set_address;
        LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].pos = LS->program_length;
        memset( &(LS->debug_line_table.statement_program[LS->program_length]), 0, 8);
          // this address size depending on target machine ???? 64-bit support?
        LS->program_length += 8;
        LS->debug_line_table.num_entry++;
    } else {
        LS->debug_line_table.statement_program[LS->program_length++] = 0; // extended opcode
        LS->debug_line_table.statement_program[LS->program_length++] = 5; // following block length
        LS->debug_line_table.statement_program[LS->program_length++] = DW_LNE_set_address;
        LS->debug_line_table.entry_pos[LS->debug_line_table.num_entry].pos = LS->program_length;
        memset( &(LS->debug_line_table.statement_program[LS->program_length]), 0, 4);
          // this address size depending on target machine ???? 64-bit support?
        LS->program_length += 4;
        LS->debug_line_table.num_entry++;
    }
    // File number should be a LEB128 number
    if (EncodeUnsignedLeb128(Table[0].file, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
        msgReport(uc2elfMsgInternalError, "when generating LEB128 number for file number");
    }

    adjustProgramBuf(LS, nbytes+1);
    LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_set_file;
    memcpy(&(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
    LS->program_length += nbytes;
    LS->lwrrent_file = Table[0].file;
    LS->is_set_file = 1;
    // generate the statement program
    for (k = 0; k < Count; k++) {
        // set file number - current file
        int line_adv;
        int addr_adv;
        Bool needCopy = True;

        if (Table[k].file == 0) {
            LS->zero_file = 1;
        }
        if (LS->lwrrent_file != Table[k].file) {
            // File number should be a LEB128 number
            if (EncodeUnsignedLeb128(Table[k].file, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
                msgReport(uc2elfMsgInternalError, "when generating LEB128 number for file number");
            }

            adjustProgramBuf(LS, nbytes+1);

            LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_set_file;
            memcpy(&(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
            LS->program_length += nbytes;
            LS->lwrrent_file = Table[k].file;
            LS->is_set_file = 1;
        } // end of is_set_file == 0

        // set current address - first time 
        if (LS->lwrrent_addr < 0) {
            if (Table[k].ucodeOff > 0) {
                // advance PC from 0 
                if (EncodeSignedLeb128(Table[k].ucodeOff, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
                    msgReport(uc2elfMsgInternalError, "when generating LEB128 number for address advance");
                }

                adjustProgramBuf(LS, nbytes+1);

                LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_pc;
                memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
                LS->program_length += nbytes;
            }

            // advance source line 
            line_adv = (int)( Table[k].line - LS->lwrrent_line);
            LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_line;
            if (EncodeSignedLeb128(line_adv, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
                msgReport(uc2elfMsgInternalError, "when generating LEB128 number for line advance");
            }

            adjustProgramBuf(LS, nbytes);
            memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
            LS->program_length += nbytes;

            // reset the current LS 
            LS->lwrrent_line = Table[k].line;
            LS->lwrrent_addr =  Table[k].ucodeOff;
            LS->lwrrent_context = 0;
            LS->lwrrent_func_address = 0;
            adjustProgramBuf(LS, 1);
        } else {
            // lwrrent_addr >= 0 
            // compute a special operator if possible 

            line_adv = Table[k].line - LS->lwrrent_line;
            addr_adv = Table[k].ucodeOff - LS->lwrrent_addr;
            if (InlineLocTable && realContext[InlineLocTable[k].Context] != LS->lwrrent_context) {
                uInt opcodeLenPos, opcodeLen = 0;
                adjustProgramBuf(LS, 3);
                LS->debug_line_table.statement_program[LS->program_length++] = 0; // extended opcode
                opcodeLenPos = LS->program_length;
                // Opcode len will be set later when actual length is available
                LS->program_length++;
                LS->debug_line_table.statement_program[LS->program_length++] = DW_LNE_inlined_call;
                opcodeLen += 1;
                if (EncodeUnsignedLeb128(realContext[InlineLocTable[k].Context],
                                         &nbytes, encode_buf, 255) == DW_DLV_ERROR)
                {
                    msgReport(uc2elfMsgInternalError, "when generating LEB128 number for setting context");
                }
                adjustProgramBuf(LS, nbytes);
                memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
                LS->program_length += nbytes;
                opcodeLen += nbytes;

                if (EncodeUnsignedLeb128(InlineLocTable[k].FunctionOffset, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
                    msgReport(uc2elfMsgInternalError, "when generating LEB128 number for setting function Offset");
                }
                adjustProgramBuf(LS, nbytes);
                memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
                LS->program_length += nbytes;
                opcodeLen += nbytes;
                LS->debug_line_table.statement_program[opcodeLenPos] = opcodeLen;

                // Flag to indicate use of opcodes related to inline function information
                LS->debug_line_table.generates_inline_info = True;
                LS->lwrrent_context = realContext[InlineLocTable[k].Context];
                LS->lwrrent_func_address = InlineLocTable[k].FunctionOffset;
            }
            if (addr_adv == 0) {
                if (line_adv == 0) {
                    needCopy = False;
                } else { // advance line only
                    LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_line;
                    if (EncodeSignedLeb128(line_adv, &nbytes, encode_buf, 255)
                            == DW_DLV_ERROR)
                    {
                        msgReport(uc2elfMsgInternalError, "when generating LEB128 number for line advance");
                    }

                    adjustProgramBuf(LS, nbytes);
                    memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
                    LS->program_length += nbytes;
                }
            }else { // addr_adv != 0
                if (line_adv == 0) { // advance addr only
                    if (addr_adv < 0) {
                          stdASSERT(0, ("negative address advance"));
                          //emit_negative_address_advance(LS, section, Table[k].ucodeOff, Is64bit ? 8 : 4);
                    } else {
                        if (EncodeSignedLeb128(addr_adv, &nbytes, encode_buf, 255) == DW_DLV_ERROR) {
                            msgReport(uc2elfMsgInternalError, "when generating LEB128 number for address advance");
                        }

                        adjustProgramBuf(LS, nbytes+1);

                        LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_pc;
                        memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes);
                        LS->program_length += nbytes;
                    }
                    // If current entry is only address advance, then we do not need
                    // to copy, But if it is marked as preseveEntry, then emit COPY opcode
                    needCopy = (InlineLocTable && InlineLocTable[k].PreserveEntry);
                } else {
                    // try special opcode
                    int sp_opc;

                    sp_opc = _dwarf_compute_special_opcode (addr_adv, line_adv,
                                                            LS->debug_line_table.first_special_opcode);
                    if (sp_opc >= 0) {
                        adjustProgramBuf(LS, 1);
                        LS->debug_line_table.statement_program[LS->program_length++] = sp_opc;
                        needCopy = False;
                        progRowId++;
                    } else { // sp_opc < 0
                        if (EncodeSignedLeb128(line_adv, &nbytes, encode_buf, 255)
                            == DW_DLV_ERROR)
                        {
                            msgReport(uc2elfMsgInternalError, "when generating LEB128 number for line advance");
                        }

                        adjustProgramBuf(LS, nbytes+1);
                        LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_line;
                        memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes); 
                        LS->program_length += nbytes; 

                        if (addr_adv < 0) {
                              stdASSERT(0, ("negative address advance"));
                              //emit_negative_address_advance(LS, section, Table[k].ucodeOff, Is64bit ? 8 : 4);
                        } else {
                            if (EncodeSignedLeb128(addr_adv, &nbytes, encode_buf, 255)
                                == DW_DLV_ERROR)
                            {
                                msgReport(uc2elfMsgInternalError, "when generating LEB128 number for address advance");
                            }

                            adjustProgramBuf(LS, nbytes+1);
                            LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_advance_pc;
                            memcpy( &(LS->debug_line_table.statement_program[LS->program_length]), encode_buf, nbytes); 
                            LS->program_length += nbytes;
                        }
                    } // sp_opc >= 0
                } // line_adv == 0
            } //  addr_adv == 0
        } // lwrrent_addr < 0

        if (needCopy) {
            progRowId++;
            LS->debug_line_table.statement_program[LS->program_length++] = DW_LNS_copy;
        }

        LS->lwrrent_file = Table[k].file;
        LS->lwrrent_addr = Table[k].ucodeOff;
        LS->lwrrent_line = Table[k].line;
        // Context is indexed from 1
        realContext[k + 1]   = progRowId;
    } // end of for 

    stdFREE(realContext);
    // end of the sequence for each function 
    adjustProgramBuf(LS, 3);
    LS->debug_line_table.statement_program[LS->program_length++] = 0; // extended opcode
    LS->debug_line_table.statement_program[LS->program_length++] = 1; // 1 byte
    LS->debug_line_table.statement_program[LS->program_length++] = 1; // end of sequence

    // reset PC address for each entry
    LS->lwrrent_addr = -1;
    LS->lwrrent_line = 1;
    LS->lwrrent_context = 0;
    LS->lwrrent_func_address = 0;
}

#define PRINT_DEBUG_INFO 0

#if PRINT_DEBUG_INFO
static void printDebugInfoMap(cString SectionName, cString FuncName, uInt Count,
                              LWuCode_DebugLineTable *Table, uInt CodeSize,
                              InlineLocInfo *InlineTable) {
  uInt Id;
  printf("#.section %s, \"\", @progbits\n", SectionName);
  printf("#.string \"%s\"\n", FuncName);
  printf("#.word %d", Count + 1);
  for (Id = 0; Id < Count + 1; ++Id) {
    printf("\n#ID %d :  %hu, %d, 0x%x(%d), %hu", (Id + 1), Table[Id].file, Table[Id].line,
                                              Table[Id].ucodeOff, Table[Id].ucodeOff,
                                              Table[Id].additionalInfo);
    if (InlineTable) {
      printf(", %hu, %llu", InlineTable[Id].Context, InlineTable[Id].FunctionOffset);
    }
  }
  if (CodeSize) {
    printf("\n Total : (0x%x) Bytes", CodeSize);
  }
  printf("\n");
}
#endif

#define IS_PROLOGUE(x) (x & DEBUG_LOC_INFO_PROLOGUE)
static uInt64 extractDebugStrOffset(String functionName)
{
   uInt64 funcOffset = 0;
   stdASSERT(stdIS_PREFIX(".debug_str+" , functionName), ("Incorrect function name"));
   String offsetPtr = strchr(functionName, '+');
   if (offsetPtr != NULL) {
       offsetPtr++;
       sscanf(offsetPtr, "%llu", &funcOffset);
   }
   return funcOffset;
}

static void addLineTable(DebugInfo *DI, 
                         LWuCodeSection_DEBUG_LINE_TABLE *LineTable,
                         LWuCodeSection_DEBUG_BLOCK_MAP *BlockMap,
                         uInt CodeSize,
                         stdMap_t BlockToLabelMap,
                         cString FuncName) {
  int Count = LineTable->count;
  LWuCode_DebugLineTable *Table;
  uInt LastSourceLine;
  uInt LastAdditionalInfo = 0;
  uInt LastIndex;
  uInt PtxLine = 0;
  uInt Offset = 0;

  if (!Count) {
      return;
  }

  stdNEW_N(Table, Count+1);
  stdMEMCOPY_N(Table, (LWuCode_DebugLineTable*)LineTable->offset.pU32, Count);
  Table[Count].ucodeOff = ~0;

  if (BlockMap) {
    LWuCode_DebugBlockMap *map = (LWuCode_DebugBlockMap*)BlockMap->offset.pU32;
    Int count = BlockMap->count;
    int aa = 0;
    int k;
    for (k = 0; k < count; k++) {
      uInt blockId = map[k].blockNo;
      Offset = map[k].ucodeOff;
      DebugLabelInfo *Label = BlockToLabelMap ?
          (DebugLabelInfo*) mapApply(BlockToLabelMap, (Pointer)(Address)blockId)
          : NULL;
      mapDefine(DI->BlockIdOffsets, (Pointer)(Address)blockId,
                                    (Pointer)(Address)Offset);
      if (Label) {
        if (!mapIsDefined(DI->SassAddresses, Label->Name)) {
          SymbolReference *Info, *PtxInfo;

          if (isDwarfEndLabel(Label->Name)) {
            PtxLine = Label->Line;
          } else {
            while (Offset >= Table[aa].ucodeOff) aa++;
            if (aa) {
              PtxLine = Table[aa-1].line;
#if 0
              if (Offset != Table[aa-1].ucodeOff) {
                stdSYSLOG("#\n# unable to recover PTX source line for label %s at address 0x%x\n#\n", Label->Name, Offset);
#if 0
                stdASSERT(False, ("unable to recover PTX source line for label %s at address 0x%x\n", Label->Name, Offset));
#endif
              }
#endif
            }
          }
          stdNEW(Info);
          Info->Entry = stdCOPYSTRING(FuncName);
          Info->Offset = Offset;
          Info->IsParam = False;
          listAddTo(Info, &DI->AllocedMemory);
          mapDefine(DI->SassAddresses, Label->Name, Info);
          if (DI->SymHandle->Mode != COMPILE_Relocatable) {
            stdNEW(PtxInfo);
            PtxInfo->Entry = Info->Entry;
            PtxInfo->Offset = PtxLine;
            PtxInfo->IsParam = False;
            listAddTo(PtxInfo, &DI->AllocedMemory);
            mapDefine(DI->LocationLabels, Label->Name, PtxInfo);
          }
        }
      }
    }
  }
  // Do we really need to remember last label for end offset?
  // Doesn't appear to matter in testing.
  Table[Count].ucodeOff = Offset > Table[Count-1].ucodeOff ? Offset
                                                            : CodeSize;
  Table[Count].line     = Table[Count-1].line;
  Table[Count].file     = Table[Count-1].file;

#if PRINT_DEBUG_INFO
  printDebugInfoMap(".lw_debug_line_sass", FuncName, Count, Table, CodeSize, NULL);
#endif
  if (DI->GetSourceInfoFromLine) {
    // if instruction map is set, must be PTX
    int k;
    Pointer LastInlineLoc;
    LWuCode_DebugLineTable entry;
    InlineLocInfo InlineLocEntry;
    stdXArray(LWuCode_DebugLineTable, LwdaSassTable);
    stdXArray(InlineLocInfo, InlineLocTable);
    stdXArray(uInt, sassAddr);
    uInt sassAddrIndex = 0;
    Bool GeneratesInlineInfo = False;
    stdMap_t ContextMap = mapNEW(Pointer, 1024); // loc => Context
    stdXArrayInit(sassAddr);
    stdXArrayInit(LwdaSassTable);
    stdXArrayInit(InlineLocTable);
    // generate LineSass

    /*
     * Lwrrently OCG provides SASS<->PTX mapping with loc.file = 0.
     * In SC, we are adding a file table entry (.lw_debug_ptx_txt.<checksum>)
     * in lw_debug_line_sass section,
     * due to which we need to set loc.file = 1 (see bug 989083).
     */
    if (DI->SymHandle->Mode == COMPILE_Relocatable) {
      int k;
      for (k = 0; k < Count; k++) Table[k].file = 1;
      Table[Count].file = Table[Count-1].file;
    }
    generateStatementProgram(DI, &DI->LineStates[SECTION_LineSass], Count+1,
                             Table, FuncName, DI->SymHandle->Is64bit, NULL);
    LastSourceLine = 0;
    LastIndex = 0;
    LastInlineLoc = NULL;
    stdVector_t InlineFileIndex = vectorCreate(8);
    stdVector_t    InlineLineNo = vectorCreate(8);
    stdVector_t          LocKey = vectorCreate(8);
    stdVector_t  InlineFuncName = vectorCreate(8);
    for (k = 0; k < Count; k++) {
      uInt ptxSourceLine = Table[k].line;
      uInt SourceLine;
      uInt SourceFile;
      uInt32 Context = 0, TestContext = 0;
      Int i;
      Bool forceEmitEntry = False;
      String functionName = NULL;
      DI->GetSourceInfoFromLine(DI->ptxInfoPtr, ptxSourceLine, &SourceLine, &SourceFile,
                                IS_PROLOGUE(Table[k].additionalInfo), &functionName,
                                InlineFileIndex, InlineLineNo, InlineFuncName, LocKey);

      // Generate debug line info only if, 
      // a non-zero loc directive is mapped to ptxSourceLine.
      if (SourceLine == 0) {
        continue;
      }

      // If current SourceLine is equal to LastSourceLine but former
      // corresponds to Prologue instruction and current to non prologue instruction
      // then as per debugger requirement we have to insert duplicate entry for
      // current SourceLine
      forceEmitEntry |= (IS_PROLOGUE(LastAdditionalInfo) &&
                        !IS_PROLOGUE(Table[k].additionalInfo));

      // If current SourceLine is equal to LastSourceLine but former
      // corresponds to different inlined_At location
      // then as insert duplicate entry for current SourceLine
      forceEmitEntry |= (vectorSize(LocKey) &&
                        (vectorIndex(LocKey, 0) != LastInlineLoc));

      if ((SourceLine != LastSourceLine) || forceEmitEntry) {
        if (vectorSize(InlineLineNo) != 0) {
          // INLINED_AT loc is associated with this location
          GeneratesInlineInfo = True;
          for (i = vectorSize(InlineLineNo) - 1; i >= 0; i--) {
            TestContext = (uInt) (Address) mapApply(ContextMap,
                                                   (Pointer) (Address) vectorIndex(LocKey, i));
            if (TestContext) {
              // INLINED_AT location is already populated, set only context
              Context = TestContext;
              LastInlineLoc = vectorIndex(LocKey, i);
            } else {
              stdXArrayAssign(sassAddr, sassAddrIndex, Table[k].ucodeOff);
              sassAddrIndex++;
              entry.file     = (uInt) (Address) vectorIndex(InlineFileIndex, i);
              entry.line     = (uInt) (Address) vectorIndex(InlineLineNo, i);
              entry.ucodeOff = ptxSourceLine;
              entry.additionalInfo = 0;
              InlineLocEntry.Context  = Context;
              // Always emit this entry in statement program as is corresponds
              // to INLINED_AT attribute
              InlineLocEntry.PreserveEntry  = True;
              InlineLocEntry.FunctionOffset = vectorIndex(InlineFuncName, i)
                                            ? extractDebugStrOffset(vectorIndex(InlineFuncName, i))
                                            : 0;
              stdXArrayAssign(LwdaSassTable,  LastIndex, entry);
              stdXArrayAssign(InlineLocTable, LastIndex, InlineLocEntry);
              Pointer inlinelocKey = vectorIndex(LocKey, i);
              LastInlineLoc = inlinelocKey;
              // Rows of Lwca-SASS tables are indexed from 1
              Context = LastIndex + 1;
              mapDefine(ContextMap, inlinelocKey, (Pointer) (Address) Context);
              LastIndex++;
            }
          }
        }
        stdXArrayAssign(sassAddr, sassAddrIndex, Table[k].ucodeOff);
        sassAddrIndex++;
        entry.file     = SourceFile;
        entry.line     = SourceLine;
        entry.ucodeOff = ptxSourceLine;
        entry.additionalInfo = Table[k].additionalInfo;
        InlineLocEntry.Context        = Context;
        InlineLocEntry.FunctionOffset = 0;
        InlineLocEntry.PreserveEntry  = False;
        if (functionName) {
          InlineLocEntry.FunctionOffset = extractDebugStrOffset(functionName);
        }
        stdXArrayAssign(LwdaSassTable,  LastIndex, entry);
        stdXArrayAssign(InlineLocTable, LastIndex, InlineLocEntry);
        LastIndex++;
      }
      LastSourceLine = SourceLine;
      LastAdditionalInfo =  Table[k].additionalInfo;
      if (vectorSize(LocKey)) {
        // Empty the inline info related vectors so they can be reused
        vectorClear(InlineFileIndex);
        vectorClear(InlineLineNo);
        vectorClear(InlineFuncName);
        vectorClear(LocKey);
      }
    }
    if (LastIndex > 0) {
      Count = LastIndex;
      entry.ucodeOff = PtxLine > LwdaSassTable[Count-1].ucodeOff ? PtxLine : LwdaSassTable[Count-1].ucodeOff;
      entry.line     = LwdaSassTable[Count-1].line;
      entry.file     = LwdaSassTable[Count-1].file;
      InlineLocEntry.Context  = InlineLocTable[Count-1].Context;
      InlineLocEntry.FunctionOffset  = InlineLocTable[Count-1].FunctionOffset;
      stdXArrayAssign(LwdaSassTable,  Count, entry);
      stdXArrayAssign(InlineLocTable, Count, InlineLocEntry);

#if PRINT_DEBUG_INFO
      printDebugInfoMap(".lw_debug_line_ptx", FuncName, Count, LwdaSassTable, 0, InlineLocTable);
#endif
      for (k = 0; k < sassAddrIndex; k++) {
        LwdaSassTable[k].ucodeOff = sassAddr[k];
      }
      entry.ucodeOff = Offset > LwdaSassTable[Count-1].ucodeOff ? Offset : CodeSize;
      entry.line     = LwdaSassTable[Count-1].line;
      entry.file     = LwdaSassTable[Count-1].file;
      stdXArrayAssign(LwdaSassTable, Count, entry);

#if PRINT_DEBUG_INFO
      printDebugInfoMap("debug_line", FuncName, Count, LwdaSassTable, 0,
                        (GeneratesInlineInfo ? InlineLocTable : NULL));
#endif
      generateStatementProgram(DI, &DI->LineStates[SECTION_Line], Count+1,
                               LwdaSassTable, FuncName, DI->SymHandle->Is64bit,
                               (GeneratesInlineInfo ? InlineLocTable : NULL));
    }
    vectorDelete(InlineFileIndex);
    vectorDelete(InlineLineNo);
    vectorDelete(InlineFuncName);
    vectorDelete(LocKey);
    stdXArrayTerm(sassAddr);
    stdXArrayTerm(InlineLocTable);
    stdXArrayTerm(LwdaSassTable);
    mapDelete(ContextMap);
  } else {
    generateStatementProgram(DI, &DI->LineStates[SECTION_Line], Count+1,
                             Table, FuncName, DI->SymHandle->Is64bit, NULL);
  }
  stdFREE(Table);
}

static void addFrameOffsetTable(DebugInfo *DI,
                                LWuCodeSection_DEBUG_FRAME_OFFSET *FrameTable,
                                LWuCodeSection_STRING_TABLE *DebugStrtab,
                                cString EntryName,
                                Int LwrFrameSize)
{

  int ii;
  cString funcname;
  LWuCode_DebugFrameOffset *debugFrameOffset =
                    (LWuCode_DebugFrameOffset *) FrameTable->offset.pU32;
  const char *strtab = (const char *)DebugStrtab->offset.pUByte;

  for (ii = 0; ii < FrameTable->count; ii++) {
      funcname = (cString)&strtab[debugFrameOffset[ii].name.offset];
      addRelocationInfo(DI->SymHandle, RELOC_DataAddress, funcname,
                        DEBUG_FRAME_SECNAME,
                        LwrFrameSize + debugFrameOffset[ii].offset,
                        0 /*addend*/);

  }

}

static void addFrameTable(DebugInfo *DI,
                          LWuCodeSection_DEBUG_FRAME_TABLE *FrameTable,
                          LWuCodeSection_DEBUG_FUNCTION_MAP *FunctionMap,
                          LWuCodeSection_STRING_TABLE *DebugStrtab,
                          cString EntryName) {
  char *frame = (char*)FrameTable->offset.pU32;
  int fdeEntries = FrameTable->fdeEntries;
  int cieEntries = FrameTable->cieEntries;
  LWuCode_DebugFunctionMap *debugFunctionMap = 
                          (LWuCode_DebugFunctionMap *)FunctionMap->offset.pU32;
  uInt debugFuncMapSize = FunctionMap->size / sizeof(*debugFunctionMap);
  uInt funcMapIndex;
  cString strtab = (cString)DebugStrtab->offset.pUByte;

  Bool is64bit = DI->SymHandle->Is64bit;
  int dwarfAddressSize = is64bit ? sizeof(uInt64) : sizeof(uInt32);
  Int ii, entrySize, sizePrinted = 0, origSize, lwrCieSize;
  ImageBuffer *debugImage;
  int value = 0;
  int dwarfWordSize = sizeof(Dwarf_Word);
  Dwarf_Word length2;
  unsigned long long ullvalue, length = 0, cie_fde_identifier = 0 ;

  // Previously ptxas processed and created elf for FrameTable in one step.
  // Now we want to process and format the FrameTable here,
  // creating a buffer and relocations that will be handled later 
  // in DebugInfoToElf.

  origSize = DI->Frame.Size; // cumulative size so far

  for (ii = 0; ii < (fdeEntries + cieEntries); ii++) {

    // The first 32b or 96b (for 32b and 64b DWARF respectively) contains the entry length excluding itself.
    if (is64bit) {
      // length
      length2 = DecodeDwarfWord((char *)(frame + sizePrinted));
      sizePrinted += dwarfWordSize;
      length = DecodeDwarfAddress((char *)(frame + sizePrinted), dwarfAddressSize);
      sizePrinted += dwarfAddressSize;
      
      entrySize = length + dwarfAddressSize + dwarfWordSize;

      // CIE_id or CIE_Pointer
      cie_fde_identifier = DecodeDwarfAddress((char *)(frame + sizePrinted), dwarfAddressSize);
      sizePrinted += dwarfAddressSize;
    } else {
      // length
      length2 = 0;
      length = DecodeDwarfWord((char *)(frame + sizePrinted));
      
      entrySize = length + dwarfWordSize;
      sizePrinted += dwarfWordSize;
      
      // CIE_id or CIE_Pointer
      Dwarf_Word temp = DecodeDwarfWord((char *)(frame + sizePrinted));
      cie_fde_identifier = temp == -1 ? -1 : (unsigned long long)temp;
      sizePrinted += dwarfWordSize;
    }

    if (cie_fde_identifier == -1) {
      // CIE is present
      
      stdNEW(debugImage);
      stdNEW_N(debugImage->Image, entrySize);
      debugImage->Size = entrySize;

      if (is64bit) {
        lwrCieSize = dwarfWordSize + 2 * dwarfAddressSize;
        // length
        stdMEMCOPY_N((Byte *)debugImage->Image, &length2, dwarfWordSize);
        stdMEMCOPY_N((Byte *)debugImage->Image + dwarfWordSize, &length, dwarfAddressSize);

        // CIE_id
        stdMEMCOPY_N((Byte *)(debugImage->Image + dwarfWordSize + dwarfAddressSize),
                     &cie_fde_identifier, dwarfAddressSize);
      } else {
        lwrCieSize = 2 * dwarfWordSize;

        // length
        stdMEMCOPY_N((Byte *)debugImage->Image, &length, dwarfWordSize);

        // CIE_id
        stdMEMCOPY_N((Byte *)(debugImage->Image + dwarfWordSize),
                     &cie_fde_identifier, dwarfWordSize);
      }

      // version
      value = (int)(*(frame + sizePrinted));
      stdMEMCOPY_N((Byte *)(debugImage->Image + lwrCieSize), &value, 1);
      sizePrinted += 1;
      lwrCieSize += 1;

      stdMEMCOPY_N((Byte *)(debugImage->Image + lwrCieSize), 
                   (char *)(frame + sizePrinted), entrySize - lwrCieSize);
      sizePrinted += entrySize - lwrCieSize;


      // append debugImage to DI->Frame
      listAddTo(debugImage, &DI->FrameBuffers);
    } else {

      //frame description entries
      cString funcname;
      ImageBuffer *entryDebugImage;
      unsigned long long offset;
      int lwrFdeSize = 0, ciePointerOffset = 0;
      uInt64 ciePointerAddend, lwrCiePointer = 0;

      stdNEW(entryDebugImage);
      stdNEW_N(entryDebugImage->Image, entrySize);
      entryDebugImage->Size = entrySize;
      // CIE pointer generated by OCG is relative to start of debug_frame section
      // of current compilation unit.However due to multiple compilation units,
      // debug_frame is appended. So make CIE pointer as relocatable.
      // Generate 0 as cie_poiner initially and populate with appropriate
      // value using relocation.
      ciePointerAddend = cie_fde_identifier + origSize;
      if (is64bit) {
        //length
        stdMEMCOPY_N((Byte *)(entryDebugImage->Image), &length2, dwarfWordSize);
        stdMEMCOPY_N((Byte *)(entryDebugImage->Image + dwarfWordSize), &length, dwarfAddressSize);

        // CIE_pointer = 0
        // Put CIE_pointer as 0 initially which will get patched with relocation
        // to populate appropriate offset
        stdMEMCOPY_N((Byte *)(entryDebugImage->Image + dwarfWordSize + dwarfAddressSize),
                     &lwrCiePointer , dwarfAddressSize);
        lwrFdeSize = dwarfWordSize + 2*dwarfAddressSize;
        ciePointerOffset = origSize + sizePrinted - dwarfAddressSize;
      } else {
        //length
        stdMEMCOPY_N((Byte *)(entryDebugImage->Image), &length, dwarfWordSize);

        // CIE_pointer = 0
        // Put CIE_pointer as 0 initially which will get patched with relocation
        // to populate appropriate offset
        stdMEMCOPY_N((Byte *)(entryDebugImage->Image + dwarfWordSize),
                     &lwrCiePointer , dwarfWordSize);
        lwrFdeSize = 2*dwarfWordSize;
        ciePointerOffset = origSize + sizePrinted - dwarfWordSize;
      }
      // Generate relocation to patch cie_pointer with .debug_frame as symbol
      // and addend as computed offset of CIE. Relocation with .debug_frame symbol
      // will ensure when sections are concatenated offset is adjusted
      // correctly in merged debug_frame section
      addRelocationInfo(DI->SymHandle, RELOC_DataAddress, DEBUG_FRAME_SECNAME,
                        DEBUG_FRAME_SECNAME,
                        ciePointerOffset,
                        ciePointerAddend  /*addend*/);
      // initial location

      // the original value is the function id of the current function,
      // we will get the offset from the table, fill it and
      // create relocations for the entry which it belongs to.
      ullvalue = DecodeDwarfAddress((char *)(frame + sizePrinted), dwarfAddressSize);

      // ullvalue represents function id however debugFrameMap is not indexed 
      // via function id.so need to explicitly search debugFunctionMap to get
      // information of required function id
      for (funcMapIndex = 0; funcMapIndex < debugFuncMapSize; ++funcMapIndex) {
        if (debugFunctionMap[funcMapIndex].funIndex == ullvalue) {
          break;
        }
      }

      stdCHECK(funcMapIndex < debugFuncMapSize, 
               (uc2elfMsgInternalError,
               "function index not found in debug function map"));

      offset = (unsigned long long)debugFunctionMap[funcMapIndex].offset;

      // initial location
      // Note offset is copied in image so relocation without addend
      // will treat offset in image as addend.
      // Offset for cloned functions will represent offset within text section
      stdMEMCOPY_N((Byte *)(entryDebugImage->Image + lwrFdeSize), &offset,
                   dwarfAddressSize);
      lwrFdeSize += dwarfAddressSize;

      funcname = (cString)&strtab[debugFunctionMap[funcMapIndex].name.offset];

      //
      // Create relocation of function name to patch initial location.
      // Initial location needs to be patched with absolute address of function.

      // In cloning case, absolute address of function will be
      // (address of entry function + offset of cloned function relative to start)
      // Relative offset of cloned function is already copied at initial location
      // So in cloning, generate relocation associated with entry function
      // symbol and offset will act as addend
      // In no-clone/SC, since each function has its own text section, generate
      // relocation based on funcname only.

      if (DI->SymHandle->Mode <= COMPILE_ExtensibleWhole &&
          !DI->SymHandle->Syscall &&
          !stdEQSTRING(EntryName, funcname)) {
          funcname = EntryName;
      }
      addRelocationInfo(DI->SymHandle, RELOC_DataAddress, funcname,
                        DEBUG_FRAME_SECNAME,
                        origSize + sizePrinted,
                        0 /*addend*/);
      sizePrinted += dwarfAddressSize;

      // address range
      ullvalue = DecodeDwarfAddress((char *)(frame + sizePrinted), dwarfAddressSize);
      stdMEMCOPY_N((Byte *)(entryDebugImage->Image + lwrFdeSize),
                   &ullvalue, dwarfAddressSize);
      addRelocationInfo(DI->SymHandle, RELOC_UnusedClear, funcname,
                        DEBUG_FRAME_SECNAME,
                        origSize + sizePrinted,
                        0 /*addend*/);
      sizePrinted += dwarfAddressSize;
      lwrFdeSize += dwarfAddressSize;
      // instructions
      stdMEMCOPY_N((Byte *)(entryDebugImage->Image + lwrFdeSize),
                   (char *)(frame + sizePrinted),
                   entrySize - lwrFdeSize);
      sizePrinted += entrySize - lwrFdeSize;

      listAddTo(entryDebugImage, &DI->FrameBuffers);
    }
  }
  DI->Frame.Size += sizePrinted;
}

#if PRINT_DEBUG_INFO
static void printSassTable(cString entry, uInt count, stdMap_t ResultIndexToSymbol,
                           LWuCode_DebugRegisterMap *map )
{
    // In addRegSassTable we generate 2 related ELF sections
    // .lw_debug_info_reg_sass and .lw_debug_info_reg_type,In this debug print
    // routine we combine data from both sections to print type information
    // along with register name in section .lw_debug_info_reg_sass
    // here we print enum value represeting PTX type
    uInt k;
    printf(".section .lw_debug_info_reg_sass + size, \"\", @progbits\n");
    printf(".string \"%s\"\n", entry);
    printf(".word %d\n", count);
    for (k = 0; k < count; k++) {
      DebugRegInfo *dregInfo = (DebugRegInfo *)mapApply(ResultIndexToSymbol,
                               (void*)(intptr_t)(map[k].resultIndex >> 4));
      String name = dregInfo->Name;
      uInt regTypeEnum = dregInfo->typeId;
      printf(".word %d .string \"%s\" .size %d\n", map[k].resultIndex & 0xf, name, regTypeEnum);
      printf(".word 0x%x, 0x%05x, 0x%05x\n",
        map[k].location, map[k].startUcodeOff, map[k].endUcodeOff);
    }
    printf("\n");

}
#endif

// In this routine we generate data for .lw_debug_info_reg_sass and
// .lw_debug_info_reg_type. .lw_debug_info_reg_type describes mapping
// of PTX and SASS registers and .lw_debug_info_reg_type represents
// type of PTX of registers present in .lw_debug_info_reg_sass
// in form of enum value
static void addRegSassTable(DebugInfo *DI,
                            LWuCodeSection_DEBUG_REGISTER_MAP *RegisterMap,
                            cString EntryName,
                            stdMap_t ResultIndexToSymbol) {
  LWuCode_DebugRegisterMap *map = 
                           (LWuCode_DebugRegisterMap*)RegisterMap->offset.pU32;
  ImageBuffer *debugImage, *debugTypeSectionImage;
  uInt32 lwrrentSize, regTypeSectionSize = 0;
  int regNameSize;
  int k;

#if PRINT_DEBUG_INFO
    printSassTable(EntryName, RegisterMap->count, ResultIndexToSymbol, map);
#endif

  // Previously ptxas processed and created elf for RegSass section in one step.
  // Now we want to process and format the RegSass table here,
  // creating a buffer that will be handled later in DebugInfoToElf.

  // To save doing multiple allocations and frees, estimate the total 
  // amount of space needed here.
  // The RegisterMap str is something like %r10;
  // will start with space for 6 chars (e.g. %rd255),
  // but realloc if more space is needed.
#define INT_SIZE 4
  regNameSize = 6;
  if (RegisterMap->count > 10000) {
    // if lots of entries, assume will need more space
    ++regNameSize; 
  }
  stdNEW(debugImage);
  debugImage->Size = strlen(EntryName) + 1 + INT_SIZE;
  lwrrentSize = debugImage->Size;
  debugImage->Size += RegisterMap->count * (regNameSize + 1 + 4 * INT_SIZE);
  stdNEW_N(debugImage->Image, debugImage->Size);
  stdMEMCOPY_N((Byte *)debugImage->Image, EntryName, strlen(EntryName) + 1);
  stdMEMCOPY_N((Byte *)(debugImage->Image + strlen(EntryName) + 1), 
               &RegisterMap->count, INT_SIZE);
  stdNEW(debugTypeSectionImage);
  debugTypeSectionImage->Size = strlen(EntryName) + 1 ;
  regTypeSectionSize = debugTypeSectionImage->Size;
  debugTypeSectionImage->Size += RegisterMap->count + INT_SIZE; // Each enum will occupy 1 byte
  stdNEW_N(debugTypeSectionImage->Image, debugTypeSectionImage->Size);
  stdMEMCOPY_N((Byte *)debugTypeSectionImage->Image, EntryName, strlen(EntryName) + 1);
  stdMEMCOPY_N((Byte *)(debugTypeSectionImage->Image + strlen(EntryName) + 1),
                        &RegisterMap->count, INT_SIZE);
  regTypeSectionSize += INT_SIZE;

  for (k = 0; k < RegisterMap->count; k++) {
    // OCG left-shits PTXAS assigned resultIndex and
    // encodes component number in lower 4 bits. So,
    // we do right-shift here. See bug comment #15 on 200153313
    // for more details.
    DebugRegInfo *dregInfo = (DebugRegInfo *)mapApply(ResultIndexToSymbol,
                                                      (void*)(intptr_t)(map[k].resultIndex >> 4));
    String str = dregInfo->Name;
    uInt regType = (uInt) dregInfo->typeId;
    stdMEMCOPY_N(debugTypeSectionImage->Image + regTypeSectionSize, &regType, 1);
    regTypeSectionSize += 1;
    Int32 index = map[k].resultIndex & 0xf;
    Int32 sassReg = map[k].location;
    Int32 sassLow = map[k].startUcodeOff;
    Int32 sassHigh = map[k].endUcodeOff;
    // need space for reg name and 4 ints
    uInt32 addSize = strlen(str) + 1 + 4 * INT_SIZE;
    Byte *p;
    if (lwrrentSize + addSize > debugImage->Size) {
      // keep increasing size as needed;
      // macros and users can use names of arbitrary length.
      while (lwrrentSize + addSize > debugImage->Size) {
        uInt32 newSize;
        ++regNameSize;
        newSize = strlen(EntryName) + 1 + INT_SIZE
                 + RegisterMap->count * (regNameSize + 1 + 4 * INT_SIZE);
        debugImage->Size = newSize;
      }
      debugImage->Image = stdREALLOC(debugImage->Image, debugImage->Size); 
    }
    p = (Byte*)debugImage->Image + lwrrentSize;

    // Encode component sent by OCG.
    stdMEMCOPY_N(p, &index, INT_SIZE);
    p += INT_SIZE;
    // Encode actual register name.
    stdMEMCOPY_N(p, str, strlen(str) + 1);
    p += strlen(str) + 1;
    // Encode SASS register info.
    stdMEMCOPY_N(p, &sassReg, INT_SIZE);
    p += INT_SIZE;
    // Encode lowest SASS adress from where the register is live.
    stdMEMCOPY_N(p, &sassLow, INT_SIZE);
    p += INT_SIZE;
    // Encode highest SASS adress where register's live range ends.
    stdMEMCOPY_N(p, &sassHigh, INT_SIZE);
    lwrrentSize += addSize;
  }

  debugImage->Size = lwrrentSize; // so allocate exact amount
  listAddTo(debugImage, &DI->RegSassBuffers);
  DI->RegSass.Size += lwrrentSize;
  debugTypeSectionImage->Size = regTypeSectionSize;
  listAddTo(debugTypeSectionImage, &DI->RegTypeBuffers);
  DI->RegType.Size += regTypeSectionSize;
}

static void addStackOffsetMap(DebugInfo *DI,
                              LWuCodeSection_DEBUG_LOCAL_VAR_MAP *LocalVarMap,
                              LWuCodeSection_STRING_TABLE *DebugStrtab,
                              cString FuncName) {
  LWuCode_DebugLocalVariableMap *map =
                      (LWuCode_DebugLocalVariableMap*)LocalVarMap->offset.pU32;
  const char *strtab = (const char *)DebugStrtab->offset.pUByte;
  Int count = LocalVarMap->count;
  Int k;

  for (k = 0; k < count; k++) {
    String varname, entryname;
    LocalVarOffset *varinfo; 
    int nameid = map[k].name.offset;
    int entryid = map[k].entry.offset; 

    /*name -- > entry+offset*/
    stdNEW_N(varname, strlen(&strtab[nameid]) + 1);
    stdMEMCOPY_N(varname, &strtab[nameid],  strlen(&strtab[nameid]) + 1);
    listAddTo(varname, &DI->AllocedMemory);

    stdNEW_N(entryname, strlen(&strtab[entryid]) + 1);
    stdMEMCOPY_N(entryname, &strtab[entryid],  strlen(&strtab[entryid]) + 1);
    listAddTo(entryname, &DI->AllocedMemory);

    stdNEW(varinfo);
    varinfo->entry = entryname;
    varinfo->regno = map[k].regNo;
    varinfo->offset = map[k].offset;
    listAddTo(varinfo, &DI->AllocedMemory);

    mapDefine(DI->StackOffsets, varname, (Pointer)varinfo);
  }
}

extern void addDebugUcodeInfo(DebugInfo *DI, CodeInfo *CI,
                              stdMap_t BlockToLabelMap,
                              stdMap_t ResultIndexToSymbolMap,
                              cString FuncName) {
  int i;
  // iterate once through sections and collect pointers to sections 
  // of interest, then go back and process data.
  LWuCodeSection_DEBUG_LINE_TABLE *LineTable = NULL;
  LWuCodeSection_DEBUG_BLOCK_MAP *BlockMap = NULL;
  LWuCodeSection_DEBUG_FRAME_TABLE *FrameTable = NULL;
  LWuCodeSection_DEBUG_FUNCTION_MAP *FunctionMap = NULL;
  LWuCodeSection_DEBUG_REGISTER_MAP *RegisterMap = NULL;
  LWuCodeSection_DEBUG_LOCAL_VAR_MAP *LocalVarMap = NULL;
  LWuCodeSection_DEBUG_FRAME_OFFSET *FrameOffsetTable = NULL;
  uInt CodeSize = 0;
  for (i=0; i < CI->UCode->header.numSections; ++i) {
    LWuCodeSectionHeader *s = &(CI->UCode->sHeader[i].genericHeader);
    switch (s->kind) {
    case LWUC_SECTION_UCODE:
      CodeSize = s->size;
      break;
    case LWUC_SECTION_DEBUG_LINE_TABLE:
      LineTable = &(CI->UCode->sHeader[i].debugLineTableHeader);
      break;
    case LWUC_SECTION_DEBUG_BLOCK_MAP:
      BlockMap = &(CI->UCode->sHeader[i].debugBlockMapHeader);
      break;
    case LWUC_SECTION_DEBUG_FRAME_TABLE:
      FrameTable = &(CI->UCode->sHeader[i].debugFrameTableHeader);
      break;
    case LWUC_SECTION_DEBUG_FRAME_OFFSET:
      FrameOffsetTable = &(CI->UCode->sHeader[i].debugFrameOffsetHeader);
      break;
    case LWUC_SECTION_DEBUG_FUNCTION_MAP:
      FunctionMap = &(CI->UCode->sHeader[i].debugFunctionMapHeader);
      break;
    case LWUC_SECTION_DEBUG_REGISTER_MAP:
      RegisterMap = &(CI->UCode->sHeader[i].debugRegisterMapHeader);
      break;
    case LWUC_SECTION_DEBUG_LOCAL_VAR_MAP:
      LocalVarMap = &(CI->UCode->sHeader[i].debugLocalVariableMapHeader);
      break;
    }
  }

  if (LineTable) {
    addLineTable(DI, LineTable, BlockMap, CodeSize, BlockToLabelMap, FuncName);
  }
  if (FrameTable && FunctionMap) {
    // there can be multiple string tables, so use one pointed to by functionMap
    LWuCodeSection_STRING_TABLE *DebugStrtab =
            &(CI->UCode->sHeader[FunctionMap->stringSection].stringTableHeader);
    if (FrameOffsetTable) {
        // Offsets generated by OCG in debug frame offset table are relative to
        // current CIE. Hence we need to add size of debug frame section generated
        // so far to make them debug_frame section relative
        Int LwrFrameSize = DI->Frame.Size;
        addFrameOffsetTable(DI, FrameOffsetTable, DebugStrtab, FuncName, LwrFrameSize);
    }
    addFrameTable(DI, FrameTable, FunctionMap, DebugStrtab, FuncName);
  }
  if (DI->GetSourceInfoFromLine) {
    // Only process some info if intermediate PTX
    if (RegisterMap) {
      addRegSassTable(DI, RegisterMap, FuncName, ResultIndexToSymbolMap);
    }
    if (LocalVarMap) {
      LWuCodeSection_STRING_TABLE *DebugStrtab =
            &(CI->UCode->sHeader[LocalVarMap->stringSection].stringTableHeader);
      addStackOffsetMap(DI, LocalVarMap, DebugStrtab, FuncName);
    }
  }
}

uInt getOffsetForBlockId(DebugInfo *DI, uInt BlockId) {
  return (uInt)(Address) mapApply(DI->BlockIdOffsets, (Pointer)(Address)BlockId);
}
// OPTIX_HAND_EDIT
#endif

DwarfSectionType dwarfSectionNameToType (String name)
{
  if (!strcmp(name, DEBUG_INFO_SECNAME)) {
    return DEBUG_INFO_SECTION;
  } else if (!strcmp(name, DEBUG_LOC_SECNAME)) {
    return DEBUG_LOC_SECTION;
  } else if (!strcmp(name, DEBUG_ABBREV_SECNAME)) {
    return DEBUG_ABBREV_SECTION;
  } else if (!strcmp(name, DEBUG_PTX_TXT_SECNAME)) {
    return DEBUG_PTX_TXT_SECTION;
  } else if (!strcmp(name, DEBUG_LINE_SECNAME)) {
    return DEBUG_LINE_SECTION;
  } else if (!strcmp(name, DEBUG_STR_SECNAME)) {
    return DEBUG_STR_SECTION;
  }
  return DEBUG_UNKNOWN_SECTION;
}
