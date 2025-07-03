/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "hal.h"
#include "prodVal.h"

void
hwprodCheckVals_GK104(FILE *prod)
{
    PRODREGDEF reg;
    PRODREGIDXDEF regIdx;
    PRODFIELDDEF field;
    BOOL regPrint;
    LwU32 regVal;
    LwU32 fieldVal;
    LwU32 mask;
    LwU32 shift;
    LwU32 i;
    LwU32 regMismatch = 0;
    LwU32 fieldMismatch = 0;
    LwU32 regMatch = 0;
    LwU32 fieldMatch = 0;
    char regName[MAX_REGNAME_LEN];
    char fieldName[MAX_FIELDNAME_LEN];

    // Start reading structure from the opened file.
    while(fread(&reg,sizeof(reg),1,prod))
    {
        // Read the register name
        if (reg.nameLen >= MAX_REGNAME_LEN)
        {
            dprintf("lw: %s string name is too large %d.\n", __FUNCTION__, reg.nameLen);
            return;
        }
        if (fread(regName, reg.nameLen, 1, prod) != 1)
        {
            dprintf("lw: %s Malformed data file, premature EOF\n", __FUNCTION__);
            return;
        }
        regName[reg.nameLen] = '\0';

        //
        // The only flags supported indicate that this is an index register.
        // Index registers require the writing of an index value to the
        // index address register before reading the register.
        //
        if (reg.flags)
        {
            if (fread(&regIdx, sizeof(regIdx), 1, prod) != 1)
            {
                dprintf("lw: %s Malformed data file, premature EOF\n", __FUNCTION__);
                return;
            }
        }

        regPrint = FALSE;
        regVal = GPU_REG_RD32(reg.addr);
        for (i = 0; i < reg.fieldCnt; i++)
        {
            if (fread(&field, sizeof(field), 1, prod) != 1)
            {
                dprintf("lw: %s Malformed data file, premature EOF\n", __FUNCTION__);
                return;
            }

            if (field.nameLen >= MAX_FIELDNAME_LEN)
            {
                dprintf("lw: Field name string is too long.\n");
                return;
            }

            // Read the field name
            if (fread(fieldName, field.nameLen, 1, prod) != 1)
            {
                dprintf("lw: %s Malformed data file, premature EOF\n", __FUNCTION__);
                return;
            }
            fieldName[field.nameLen] = '\0';

            if (reg.flags)
            {
                // For now skip all index registers because they are in dev_disp
                // anyway and really shouldn't have prod values.
                continue;
            }

            // Get the value of the field using the field description.
            mask = DRF_MASK((field.endBit):(field.startBit));
            shift = DRF_SHIFT((field.endBit):(field.startBit));
            fieldVal = (regVal >> shift) & mask;

            // Check for a mismatch
            if (field.value != fieldVal)
            {
                if (!regPrint)
                {
                    dprintf(
                        "Register %s (0x%08x) mismatch\n", regName, reg.addr);
                    regPrint = TRUE;
                    regMismatch++;
                }
                dprintf(
                    "   Field %s (%d:%d) expected: 0x%04x read: 0x%04x\n",
                     fieldName,
                     field.startBit,
                     field.endBit,
                     field.value,
                     fieldVal);

                fieldMismatch++;
            }
            else
            {
                fieldMatch++;
            }
        }
        if (!regPrint)
        {
            regMatch++;
        }
    }

    if(!feof(prod))
    {
        dprintf("lw: %s Malformed data file, premature EOF\n", __FUNCTION__);
        return;
    }

    fclose(prod);

    if (regMismatch != 0)
    {
        dprintf("Register Matches: %d\n", regMatch);
        dprintf("Field Matches: %d\n", fieldMatch);
        dprintf("\n");
        dprintf("Register Mismatches: %d\n", regMismatch);
        dprintf("Field Mismatches: %d\n", fieldMismatch);
    }
}

