/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   unitodb.c
 * @brief  functions to get/set pdb property and init the odbcommon structure
 */

#include "unitodb.h"
#include "utility.h"

/*!
 * @brief initialize a common object data base structure's function pointers.
 *
 * @param[in]      thisCommon   pointer to the common object data base
 *
 * @param[in]      propBegin    start of property range
 *
 * @param[in]      propEnd      end of the property range
 *
 */
static void
unitOdbInitProperties(PODBCOMMON thisCommon, LwU32 propBegin, LwU32 propEnd)
{
    //
    // we check that both constants are from the same object
    // this check will also fail if the number of props is exceeded
    //
    UNIT_ASSERT(OBJECT_EQUAL_PROPS(propBegin,propEnd));

    // we check that function params are in correct order
    UNIT_ASSERT(propBegin < propEnd);

    thisCommon->propRangeBegin = propBegin;
    thisCommon->propRangeEnd   = propEnd;
}

/*!
 * @brief Set a boolean property bit.
 *
 * @param[in]      thisPdbData  pointer PDB data
 *
 * @param[in]      prop         actual property value being added
 *
 * @param[in]      propValue    a 0 or non-0 value for a particular bit
 *
 */
static void
unitPdbSetProperty(PPDBDATA thisPdbData, LwU32 prop,
               BOOL propValue, LwU32 propRangeStart)
{
    LwU32 elementArrayIndex = (prop - propRangeStart) / PDB_ELEMENT_BIT_SIZE;
    LwU32 elementBitIndex = (prop - propRangeStart) % PDB_ELEMENT_BIT_SIZE;

    if (elementArrayIndex >= MAX_PDB_ELEMENTS)
    {
        UNIT_ASSERT(elementArrayIndex < MAX_PDB_ELEMENTS);
    }

    thisPdbData->PdbElements[elementArrayIndex] &= ~(1 << elementBitIndex);
    thisPdbData->PdbElements[elementArrayIndex] |=
                                                (propValue << elementBitIndex);
}

/*!
 * @brief Determine if a given property is set in the current
 *        properties database.
 *
 * @param[in]      thisPdbData     pointer PDB data
 *
 * @param[in]      prop            actual property value being added
 *
 * @param[in]      propRangeStart  start of the property range
 *
 */
static BOOL
 unitPdbGetProperty(PPDBDATA thisPdbData, LwU32 prop, LwU32 propRangeStart)
{
    LwU32 elementArrayIndex = (prop - propRangeStart) / PDB_ELEMENT_BIT_SIZE;
    LwU32 elementBitIndex = (prop - propRangeStart) % PDB_ELEMENT_BIT_SIZE;

    if (elementArrayIndex >= MAX_PDB_ELEMENTS)
    {
        UNIT_ASSERT(elementArrayIndex < MAX_PDB_ELEMENTS);
        return FALSE;
    }

    // force value to TRUE/FALSE from bitmask
    return !!(thisPdbData->PdbElements[elementArrayIndex] &
             (1 << elementBitIndex));
}

/*!
 * @brief Set all the properties passed in to TRUE. This function is
 *        used by the rmconfig generated code to set a list of
 *        properties
 *
 * @param[in]      thisPdbData      pointer PDB data
 *
 * @param[in]      thisPropertyList properties to be set to TRUE
 *
 * @param[in]      thisNum          number of properties in the list
 *
 */
void
odbSetProperties
(
    PODBCOMMON thisCommon,
    PDB_PROP_BASE_TYPE* thisPropertyList,
    LwU32 thisNum
)
{
    LwU32 index;
    // Set the property value in the properties database for this object.
    for (index = 0; index < thisNum; index++)
    {
        UNIT_ASSERT(((LwU32)thisPropertyList[index] >
                    thisCommon->propRangeBegin) &&
                    ((LwU32)thisPropertyList[index] < thisCommon->propRangeEnd));
        unitPdbSetProperty(&(thisCommon->propDatabase), thisPropertyList[index],
                       TRUE, thisCommon->propRangeBegin);
    }
}

/*!
 * @brief Report the pointer to the PDB database property bits array
 *        associated with a common object data base structure.
 *
 * @param[in]      thisCommon   pointer to the common object data base
 *
 * @param[in]      thisProperty property to be retrieved
 *
 * @return         BOOL         TRUE/FALSE
 *
 */
static BOOL
odbGetProperty(PODBCOMMON thisCommon, LwU32 thisProperty)
{
    // Return the property value found in the properties database for this object.
    UNIT_ASSERT((thisProperty > thisCommon->propRangeBegin)
                && (thisProperty < thisCommon->propRangeEnd));
    return (unitPdbGetProperty(&(thisCommon->propDatabase), thisProperty,
                          thisCommon->propRangeBegin));
}

/*!
 * @brief Report the pointer to the PDB database property bits array
 *        associated with a common object data base structure.
 *
 * @param[in]      thisCommon   pointer to the common object data base
 *
 * @param[in]      thisProperty property to be retrieved
 *
 * @param[in]      thisValue    value to be set into the property
 *
 */
static void
odbSetProperty(PODBCOMMON thisCommon, LwU32 thisProperty, BOOL thisValue)
{
    // Set the property value in the properties database for this object.
    UNIT_ASSERT((thisProperty > thisCommon->propRangeBegin)
                 && (thisProperty < thisCommon->propRangeEnd));
    unitPdbSetProperty(&(thisCommon->propDatabase), thisProperty,
                  thisValue, thisCommon->propRangeBegin);
}

/*!
 * @brief Initialize a common object data base structure's function pointers.
 *
 * @param[in]      thisCommon   pointer to the common object data base
 */
static void
unitOdbInitCommonFunctionPointers(PODBCOMMON thisOdbCommon)
{
    thisOdbCommon->odbGetProperty    = odbGetProperty;
    thisOdbCommon->odbSetProperty    = odbSetProperty;
    thisOdbCommon->odbSetProperties  = odbSetProperties;
    thisOdbCommon->odbInitProperties = unitOdbInitProperties;
}

/*!
 * @brief Initialize the properties database elements to zero.
 *
 * @param[in]      thisPdbData      pointer PDB data
 */
static void
unitPdbClear(PPDBDATA thisPdbData)
{
    memset(thisPdbData, 0, sizeof(PDBDATA));
}

/*!
 * @brief Initialize a common object data base structure.
 *
 * @param[in]      thisCommon   pointer to the common object data base
 */
void unitOdbInitCommon(PODBCOMMON thisOdbCommon)
{
    // Null out the properties database.
    unitPdbClear(&(thisOdbCommon->propDatabase));

    // Initialize the function pointers.
    unitOdbInitCommonFunctionPointers(thisOdbCommon);
}
