/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  seqdump.c
 * @brief WinDbg Extension for the PMU Sequencer
 */

#include "os.h"
#include "pmu/pmuseqinst.h"

#define OPC(opc)                                          LW_PMU_SEQ_##opc##_OPC

static const struct
{
    LwU8        opcode;
    const char *pName;
} SeqOpcodeTable[] =
{
    {OPC(LOAD_ACC)            , "LOAD_ACC"            },
    {OPC(LOAD_ADDR)           , "LOAD_ADDR"           },
    {OPC(OR_ADDR)             , "OR_ADDR"             },
    {OPC(AND_ADDR)            , "AND_ADDR"            },
    {OPC(ADD_ACC)             , "ADD_ACC"             },
    {OPC(ADD_ADDR)            , "ADD_ADDR"            },
    {OPC(LSHIFT_ADDR)         , "LSHIFT_ADDR"         },
    {OPC(READ_ADDR)           , "READ_ADDR"           },
    {OPC(READ_IMM)            , "READ_IMM"            },
    {OPC(READ_INDEX)          , "READ_INDEX"          },
    {OPC(WRITE_IMM)           , "WRITE_IMM"           },
    {OPC(WRITE_INDEX)         , "WRITE_INDEX"         },
    {OPC(WAIT_NS)             , "WAIT_NS"             },
    {OPC(WAIT_SIGNAL)         , "WAIT_SIGNAL"         },
    {OPC(POLL_CONTINUE)       , "POLL_CONTINUE"       },
    {OPC(EXIT)                , "EXIT"                },
    {OPC(CMP)                 , "CMP"                 },
    {OPC(BREQ)                , "BREQ"                },
    {OPC(BRNEQ)               , "BRNEQ"               },
    {OPC(BRLT)                , "BRLT"                },
    {OPC(BRGT)                , "BRGT"                },
    {OPC(BRA)                 , "BRA"                 },
    {OPC(ENTER_CRITICAL)      , "ENTER_CRITICAL"      },
    {OPC(EXIT_CRITICAL)       , "EXIT_CRITICAL"       },
    {OPC(FB_STOP)             , "FB_STOP"             },
    {OPC(WRITE_REG)           , "WRITE_REG"           },
    {OPC(LOAD_VREG)           , "LOAD_VREG"           },
    {OPC(LOAD_VREG_IND)       , "LOAD_VREG_IND"       },
    {OPC(LOAD_VREG_VAL)       , "LOAD_VREG_VAL"       },
    {OPC(LOAD_VREG_VAL_IND)   , "LOAD_VREG_VAL_IND"   },
    {OPC(LOAD_ADDR_VREG)      , "LOAD_ADDR_VREG"      },
    {OPC(LOAD_ADDR_VREG_IND)  , "LOAD_ADDR_VREG_IND"  },
    {OPC(ADD_VREG)            , "ADD_VREG"            },
    {OPC(CMP_VREG)            , "CMP_VREG"            },
    {OPC(WAIT_FLUSH_NS)       , "WAIT_FLUSH_NS"       },
    {OPC(LOAD_VREG_TS_NS)     , "LOAD_VREG_TS_NS"     },
    {OPC(LOAD_VREG_TS_NS_IND) , "LOAD_VREG_TS_NS_IND" },
    {OPC(DMA_READ_NEXT_BLOCK) , "DMA_READ_NEXT_BLOCK" },
};

LwBool
dumpInstruction
(
    LwU8         opcode,
    LwU8         size,
    const LwU32 *pData
)
{
    LwU32  i;
    LwU32  numElements;

    numElements = LW_ARRAY_ELEMENTS(SeqOpcodeTable);
    for (i = 0; i < numElements; i++)
    {
        if (SeqOpcodeTable[i].opcode == opcode)
        {
            break;
        }
    }
    if (i >= numElements)
    {
        return LW_FALSE;
    }

    dprintf("%-20s : ", SeqOpcodeTable[i].pName);
    for (i = 0; i < size; i++)
    {
        if ((i != 0) && (i % 4) == 0)
        {
            dprintf("\n%-20s : ", "");
        }
        dprintf("0x%08x ", pData[i]);
    }
    dprintf("\n");

    return LW_TRUE;
}

void
seqDumpScript
(
    LwU32 *pScript,
    LwU32  sizeBytes
)
{
    LwU32   elements = sizeBytes / sizeof(LwU32);
    LwU8    opcode;
    LwU8    size;
    LwBool  bDone = LW_FALSE;
    LwBool  bValid;
    LwU32  *pScriptStart = pScript;
    LwU32  *pScriptEnd   = &pScript[elements];

    while (!bDone)
    {
        opcode  = (LwU8)(*pScript >> LW_PMU_SEQ_INSTR_OPC_SHIFT);
        opcode &= LW_PMU_SEQ_INSTR_OPC_MASK;
        size    = (LwU8)(*pScript >> LW_PMU_SEQ_INSTR_SIZE_SHIFT);
        size   &= LW_PMU_SEQ_INSTR_SIZE_MASK;

        // validate instruction size
        if ((size == 0) || ((pScript + size) > pScriptEnd))
        {
            bValid = LW_FALSE;
            break;
        }

        bValid = dumpInstruction(opcode, size, pScript);
        if (bValid)
        {
            pScript += size;
            if (pScript >= pScriptEnd)
            {
                bDone = LW_TRUE;
            }
        }
        else
        {
            break;
        }
    }

    if (!bValid)
    {
        dprintf("Sequencer script is malformed. Bailing\n");
        dprintf("Failed at offset: %d\n", (LwU32)(pScript - pScriptStart));
        dprintf("            Data: 0x%08x\n", *pScript);
    }
}

