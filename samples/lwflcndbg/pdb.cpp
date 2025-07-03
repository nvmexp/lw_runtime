/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2012 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// ttsu@lwpu.com - July 2008
// pdb.cpp - module for reading property database data
//
//*****************************************************

#include <string.h>

#include "os.h"
#include "CSymHelpers.h"
#include "CSymModule.h"
#include "CSymType.h"
#include "pdb.h"

// The bit size of each element in the propValues array
#define PDB_ELEMENT_BIT_SIZE (sizeof(ULONG) * 8)

//
// pdbEnumFromOdbClass()
//  - Obtains the name of the property database enumeration given object class
//  Returns :   Name of the property database enumeration type, or NULL
//                  NOTE: This function allocates memory which must be freed.
//  Params  :   odbClassName - name of the object class of interest
//
char * pdbEnumFromOdbClass(
    const char* odbClassName
)
{
    size_t odbClassLen = strlen(odbClassName);
    char * pdbEnumName = (odbClassLen < sizeof("ODB_CLASS_") ? NULL
                                                             : new char[strlen(odbClassName)]);
                                                             
    if (NULL == pdbEnumName)
        return NULL;
    
    // do a string replacement: ODB_CLASS_ => PDB_PROP_
    strcpy(pdbEnumName, "PDB_PROP_");
    // CORELOGIC Class uses CL abbreviation in PDB properties
    if (strcmp(odbClassName, "ODB_CLASS_CORELOGIC") == 0)
    {
        strcat(pdbEnumName, "CL");
    }
    else
    {
        strcat(pdbEnumName, (odbClassName + sizeof("ODB_CLASS_") - 1));
    }
    
    return pdbEnumName;
}

//
// pdbDump()
//  - Outputs all of the properties from the specified object
//  Returns :   
//  Params  :   rootObjAddr - address of the object containing properties
//              propEnumName - name of the enumeration for the object's pdb
//                  For example: "PDB_PROP_XXXX"
//  Preconditions: ExtQuery() and InitGlobals() must have been called
//              prior to this function
//                  The properties have been initialized
//
void
pdbDump(
    ULONG64     rootObjAddr, 
    const char* propEnumName
)
{
    CSymModule  kmdModule(g_KMDModuleName);
    CSymType    odbCommon(&kmdModule, "ODBCOMMON");
    CSymType    pdbData(&kmdModule, "PDBDATA");
    CSymType    pdbKeyEnum(&kmdModule, propEnumName);
    
    PULONG      propValues;
    ULONG       propBegin;
    ULONG       propEnd;
    char        propName[64];
    BOOL        propVal;
    // getSize() returns the total number of bytes in PDBDATA struct
    ULONG       pdbNumElements = (pdbData.getSize() + sizeof(ULONG) - 1) / sizeof(ULONG);
    ULONG       elementArrayIndex;
    ULONG       elementBitIndex;

    propValues = new ULONG[pdbNumElements];
    
    odbCommon.read(rootObjAddr, "propDatabase", propValues, pdbData.getSize());
    propBegin = odbCommon.readULONG(rootObjAddr, "propRangeBegin");
    propEnd = odbCommon.readULONG(rootObjAddr, "propRangeEnd");

    if (propBegin < (propEnd - 1))
    {
        for (ULONG propIndex = (propBegin + 1); propIndex < propEnd; ++propIndex)
        {
            elementArrayIndex = (propIndex - propBegin) / PDB_ELEMENT_BIT_SIZE;
            elementBitIndex = (propIndex - propBegin) % PDB_ELEMENT_BIT_SIZE;
            
            // retrieve the name of the property
            pdbKeyEnum.getConstantName(propIndex, propName, sizeof(propName));
            
            // find the array index that contains the given property
            propVal = propValues[elementArrayIndex] & (1 << elementBitIndex);
            dprintf("  %-50s: %d\n", propName, (propVal ? 1 : 0));
        }
    }
    else
    {
        dprintf("  [no properties in this object class]\n");
    }

    delete [] propValues;
}
