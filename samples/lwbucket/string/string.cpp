 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: string.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#define _CRT_SELWRE_NO_WARNINGS
#include "../include/string.h"

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
static  void*           stringAlloc(size_t size);
static  void*           stringRealloc(void* pString, size_t size);
static  void            stringFree(void* pString);

//******************************************************************************

CString::CString()
:   m_pString(NULL)
{
    // Allocate enough space for a NULL string
    m_pString = charptr(stringAlloc(1));
    if (m_pString != NULL)
    {
        // NULL terminate the string
        m_pString[0] = '\0';
    }
    else    // String allocation error
    {
        // Throw memory error
        throw CMemoryException(1, __FILE__, __FUNCTION__, __LINE__);
    }

} // CString

//******************************************************************************

CString::CString
(
    const CString&      sString,
    size_t              position,
    size_t              length
)
:   m_pString(NULL)
{
    size_t              allocated;

    // Check for an invalid position
    if (position > sString.length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d",
                               position, sString.length());
    }
    // Compute the amount of space required (Should be at least one byte for terminator)
    allocated = strnlen(&sString.m_pString[position], min(length, (sString.capacity() - position))) + 1;

    // Try to allocate space for string buffer
    m_pString = charptr(stringAlloc(allocated));
    if (m_pString != NULL)
    {
        // Copy string data from given string
        memcpy(m_pString, &sString.m_pString[position], (allocated - 1));

        // Terminate the string
        m_pString[allocated - 1] = '\0';
    }
    else    // String allocation error
    {
        // Throw memory error
        throw CMemoryException(allocated, __FILE__, __FUNCTION__, __LINE__);
    }

} // CString

//******************************************************************************

CString::CString
(
    const char         *pString,
    size_t              length
)
:   m_pString(NULL)
{
    size_t              allocated = 1;

    // Compute the amount of space required
    if (pString != NULL)
    {
        allocated = strnlen(pString, length) + 1;
    }
    // Try to allocate space for string buffer
    m_pString = charptr(stringAlloc(allocated));
    if (m_pString != NULL)
    {
        // Copy string data from given string
        memcpy(m_pString, pString, (allocated - 1));

        // Terminate the string
        m_pString[allocated - 1] = '\0';
    }
    else    // String allocation error
    {
        // Throw memory error
        throw CMemoryException(allocated, __FILE__, __FUNCTION__, __LINE__);
    }

} // CString

//******************************************************************************

CString::CString
(
    size_t              length,
    char                c
)
:   m_pString(NULL)
{
    size_t              allocated = length + 1;

    // Try to allocate the requested space
    m_pString = charptr(stringAlloc(allocated));
    if (m_pString != NULL)
    {
        // Initialize the string contents
        memset(m_pString, c, length);

        // Terminate the string
        m_pString[length] = '\0';
    }
    else    // String allocation error
    {
        // Throw memory error
        throw CMemoryException(allocated, __FILE__, __FUNCTION__, __LINE__);
    }

} // CString

//******************************************************************************

CString::~CString()
{
    // Check for string buffer allocated (Should always be)
    if (m_pString != NULL)
    {
        // Free the string buffer
        stringFree(m_pString);
    }

} // ~CString

//******************************************************************************

CString&
CString::append
(
    const CString&      sString,
    size_t              position,
    size_t              length
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Check for an invalid position
    if (position > sString.length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, sString.length());
    }
    // Compute the amount of additional space required
    amount = strnlen(&sString.m_pString[position], min(length, (sString.capacity() - position)));

    // Get the current string size [length] (Append position)
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to append to current string
    if (amount != 0)
    {
        // Append the given string to the end of the current string
        memcpy(&m_pString[current], &sString.m_pString[position], amount);

        // Terminate the string
        m_pString[current + amount] = '\0';
    }
    return *this;

} // append

//******************************************************************************

CString&
CString::append
(
    const char         *pString,
    size_t              position,
    size_t              length
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Compute the amount of space required
    if (pString != NULL)
    {
        amount = strnlen(&pString[position], length);
    }
    // Get the current string size [length] (Append position)
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to append to current string
    if (amount != 0)
    {
        // Append the given string to the end of the current string
        memcpy(&m_pString[current], &pString[position], amount);

        // Terminate the string
        m_pString[current + amount] = '\0';
    }
    return *this;

} // append

//******************************************************************************

CString&
CString::append
(
    size_t              length,
    char                c
)
{
    size_t              current;
    size_t              space;

    // Get the current string size [length] (Append position)
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + length) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to append to current string
    if (length != 0)
    {
        // Append given character to the end of the current string
        memset(&m_pString[current], c, length);

        // Terminate the string
        m_pString[current + length] = '\0';
    }
    return *this;

} // append

//******************************************************************************

CString&
CString::assign
(
    const CString&      sString,
    size_t              position,
    size_t              length
)
{
    size_t              space;
    size_t              amount = 0;

    // Check for an invalid position
    if (position > sString.length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, sString.length());
    }
    // Compute the amount of space required
    amount = strnlen(&sString.m_pString[position], min(length, (sString.capacity() - position))) + 1;

    // Compute amount of space that needs to be allocated
    space = capacity();
    while (amount >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to assign to current string
    if (amount != 0)
    {
        // Assign the given string to the current string
        memcpy(m_pString, &sString.m_pString[position], (amount - 1));

        // Terminate the string
        m_pString[amount - 1] = '\0';
    }
    return *this;

} // assign

//******************************************************************************

CString&
CString::assign
(
    const char         *pString,
    size_t              position,
    size_t              length
)
{
    size_t              space;
    size_t              amount = 0;

    // Compute the amount of space required
    if (pString != NULL)
    {
        amount = strnlen(&pString[position], length) + 1;
    }
    // Compute amount of space that needs to be allocated
    space = capacity();
    while (amount >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to assign to current string
    if (amount != 0)
    {
        // Assign the given string to the current string
        memcpy(m_pString, &pString[position], (amount - 1));

        // Terminate the string
        m_pString[amount - 1] = '\0';
    }
    return *this;

} // assign

//******************************************************************************

CString&
CString::assign
(
    size_t              length,
    char                c
)
{
    size_t              space;
    size_t              amount;

    // Compute the amount of space required
    amount = length + 1;

    // Compute amount of space that needs to be allocated
    space = capacity();
    while (amount >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Assign given character to the current string (for requested length)
    memset(m_pString, c, length);

    // Terminate the string
    m_pString[amount - 1] = '\0';

    return *this;

} // assign

//******************************************************************************

char
CString::at
(
    size_t              position
) const
{
    // Check for an invalid position
    if (position > length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, length());
    }
    return m_pString[position];

} // at

//******************************************************************************

size_t
CString::capacity() const
{
    STRING_DATA        *pStringData = reinterpret_cast<STRING_DATA*>(m_pString - sizeof(STRING_DATA));

    // Return the amount of allocated space (Capacity)
    return pStringData->allocated;

} // capacity

//******************************************************************************

void
CString::clear()
{
    // Free any allocated space (Should always be some)
    if (m_pString != NULL)
    {
        stringFree(m_pString);
    }
    // Allocate enough space for a NULL string
    m_pString = charptr(stringAlloc(1));
    if (m_pString != NULL)
    {
        // NULL terminate the string
        m_pString[0] = '\0';
    }
    else    // String allocation error
    {
        // Throw memory error
        throw CMemoryException(1, __FILE__, __FUNCTION__, __LINE__);
    }

} // clear

//******************************************************************************

int
CString::compare
(
    const CString&      sString
) const
{
    size_t              compare;
    size_t              compare1;
    size_t              compare2;

    // Compute the comparison lengths
    compare1 = length();
    compare2 = sString.length();

    // Get the actual comparison length (Minimum of comparison lengths)
    compare = min(compare1, compare2);

    // Check for strings to compare
    if (compare != 0)
    {
        return strncmp(m_pString, sString.m_pString, compare);
    }
    else    // Only one string, compare lengths
    {
        return static_cast<int>((compare1 - compare2));
    }

} // compare

//******************************************************************************

int
CString::compare
(
    const char         *pString
) const
{
    size_t              compare;
    size_t              compare1;
    size_t              compare2 = 0;

    // Compute the comparison lengths
    compare1 = length();
    if (pString != NULL)
    {
        compare2 = strlen(pString);
    }
    // Get the actual comparison length (Minimum of comparison lengths)
    compare = min(compare1, compare2);

    // Check for strings to compare
    if (compare != 0)
    {
        return strncmp(m_pString, pString, compare);
    }
    else    // Only one string, compare lengths
    {
        return static_cast<int>((compare1 - compare2));
    }

} // compare

//******************************************************************************

int
CString::compare
(
    size_t              position1,
    size_t              length1,
    const CString&      sString,
    size_t              position2,
    size_t              length2
) const
{
    size_t              compare;
    size_t              compare1 = 0;
    size_t              compare2 = 0;

    // Check for invalid position(s)
    if (position1 > length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, length());
    }
    if (position2 > sString.length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position2, sString.length());
    }
    // Compute the comparison lengths
    if (m_pString != NULL)
    {
        compare1 = strnlen(&m_pString[position1], min(length1, (capacity() - position1)));
    }
    if (sString.m_pString != NULL)
    {
        compare2 = strnlen(&sString.m_pString[position2], min(length2, (sString.capacity() - position2)));
    }
    // Get the actual comparison length (Minimum of comparison lengths)
    compare = min(compare1, compare2);

    // Check for strings to compare
    if (compare != 0)
    {
        return strncmp(&m_pString[position1], &sString.m_pString[position2], compare);
    }
    else    // Only one string, compare lengths
    {
        return static_cast<int>((compare1 - compare2));
    }

} // compare

//******************************************************************************

int
CString::compare
(
    size_t              position1,
    size_t              length1,
    const char         *pString,
    size_t              position2,
    size_t              length2
) const
{
    size_t              compare;
    size_t              compare1 = 0;
    size_t              compare2 = 0;

    // Check for an invalid position
    if (position1 > length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d",
                               position1, length());
    }
    // Compute the comparison lengths
    if (m_pString != NULL)
    {
        compare1 = strnlen(&m_pString[position1], min(length1, (capacity() - position1)));
    }
    if (pString != NULL)
    {
        compare2 = min(strlen(&pString[position2]), length2);
    }
    // Get the actual comparison length (Minimum of comparison lengths)
    compare = min(compare1, compare2);

    // Check for strings to compare
    if (compare != 0)
    {
        return strncmp(&m_pString[position1], &pString[position2], compare);
    }
    else    // Only one string, compare lengths
    {
        return static_cast<int>((compare1 - compare2));
    }

} // compare

//******************************************************************************

size_t
CString::copy
(
    char               *pString,
    size_t              length,
    size_t              position
)
{
    size_t              amount = 0;

    // Check for an invalid position
    if (position > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d",
                               position, size());
    }
    // Compute the amount to copy
    amount = strnlen(&m_pString[position], min(length, (capacity() - position)));

    // Check for data to copy
    if (amount != 0)
    {
        // Check for location to copy data
        if (pString != NULL)
        {
            memcpy(pString, &m_pString[position], amount);
        }
    }
    return amount;

} // copy

//******************************************************************************

char*
CString::data
(
    size_t              position
)
{
    // Check for an invalid position
    if (position > length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, length());
    }
    return &m_pString[position];

} // data

//******************************************************************************

CString&
CString::erase
(
    size_t              position,
    size_t              length
)
{
    size_t              current;
    size_t              where;
    size_t              amount = 0;

    // Get the current string size [length]
    current = size();

    // Check for an invalid position
    if (position > current)
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, current);
    }
    // Check for something to erase
    if (length != 0)
    {
        // Compute where to move from and the amount to move 
        where  = min(length, current - position);
        amount = current - where;

        // Copy amount to move from computed location (Erasing the requested length)
        memmove(&m_pString[position], &m_pString[position + where], (amount + 1));
    }
    return *this;

} // erase

//******************************************************************************

CString&
CString::fill
(
    char                c,
    size_t              length
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Get the current string size [length]
    current = size();

    // Compute the amount to be filled
    amount = max(length, current);

    // Compute amount of space that needs to be allocated
    space = capacity();
    while (amount >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to fill in the current string
    if (amount != 0)
    {
        // Fill string with requested character for required amount
        memset(m_pString, c, amount);
    }
    // Check for termination required
    if (amount >= current)
    {
        // Terminate the string
        m_pString[amount] = '\0';
    }
    return *this;

} // fill

//******************************************************************************

CString&
CString::fill
(
    size_t              position,
    size_t              length,
    char                c
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Get the current string size [length]
    current = size();

    // Check for an invalid position
    if (position > current)
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, current);
    }
    // Compute the amount to be filled
    amount = max(length, current - position);

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount - position) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to fill in the current string
    if (amount != 0)
    {
        // Fill string with requested character at requested position for length
        memset(&m_pString[position], c, amount);
    }
    // Check for termination required
    if ((position + amount) >= current)
    {
        // Terminate the string
        m_pString[position + amount] = '\0';
    }
    return *this;

} // fill

//******************************************************************************

size_t
CString::find
(
    const CString&      sString,
    size_t              position,
    size_t              length
) const
{
    CString             sTemporary;
    char               *pLocation;
    size_t              amount   = 0;
    size_t              location = NOT_FOUND;

    // Check for an invalid position
    if (position > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, size());
    }
    // Compute the maximum amount to search
    amount = strnlen(&m_pString[position], min(length, (capacity() - position)));

    // Check for something to search
    if (amount != 0)
    {
        // Check for need to limit the search
        if (amount == size())
        {
            // Try to find the given string in this string
            pLocation = strstr(&m_pString[position], sString.m_pString);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - m_pString) + position;
            }
        }
        else    // Need to limit search
        {
            // Create a temporary shortened copy of the string
            sTemporary.assign(*this, position, amount);

            // Try to find the given string in this temporary string
            pLocation = strstr(sTemporary.m_pString, sString.m_pString);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - sTemporary.m_pString) + position;
            }
        }
    }
    return location;

} // find

//******************************************************************************

size_t
CString::find
(
    const char         *pString,
    size_t              position,
    size_t              length
) const
{
    CString             sTemporary;
    char               *pLocation;
    size_t              amount   = 0;
    size_t              location = NOT_FOUND;

    // Check for an invalid position
    if (position > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, size());
    }
    // Compute the maximum amount to search
    amount = strnlen(&m_pString[position], min(length, (capacity() - position)));

    // Check for something to search
    if (amount != 0)
    {
        // Check for need to limit the search
        if (amount == size())
        {
            // Try to find the given string in this string
            pLocation = strstr(&m_pString[position], pString);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - m_pString) + position;
            }
        }
        else    // Need to limit search
        {
            // Create a temporary shortened copy of the string
            sTemporary.assign(*this, position, amount);

            // Try to find the given string in this temporary string
            pLocation = strstr(sTemporary.m_pString, pString);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - sTemporary.m_pString) + position;
            }
        }
    }
    return location;

} // find

//******************************************************************************

size_t
CString::find
(
    char                c,
    size_t              position,
    size_t              length
) const
{
    CString             sTemporary;
    char               *pLocation;
    size_t              amount   = 0;
    size_t              location = NOT_FOUND;

    // Check for an invalid position
    if (position > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, size());
    }
    // Compute the maximum amount to search
    amount = strnlen(&m_pString[position], min(length, (capacity() - position)));

    // Check for something to search
    if (amount != 0)
    {
        // Check for need to limit the search
        if (amount == size())
        {
            // Try to find the given character in this string
            pLocation = strchr(&m_pString[position], c);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - m_pString) + position;
            }
        }
        else    // Need to limit search
        {
            // Create a temporary shortened copy of the string
            sTemporary.assign(*this, position, amount);

            // Try to find the given character in this temporary string
            pLocation = strchr(sTemporary.m_pString, c);

            // Check for string located
            if (pLocation != NULL)
            {
                // Compute the location of the match (offset)
                location = (pLocation - sTemporary.m_pString) + position;
            }
        }
    }
    return location;

} // find

//******************************************************************************

CString&
CString::insert
(
    size_t              position1,
    const CString&      sString,
    size_t              position2,
    size_t              length
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Check for invalid positions
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    if (position2 > sString.size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position2, sString.size());
    }
    // Compute the amount of additional space required
    amount = strnlen(&sString.m_pString[position2], min(length, (sString.capacity() - position2)));

    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to insert in the current string
    if (amount != 0)
    {
        // Check for data to move before insertion
        if (position1 != size())
        {
            // Move current string data to make space for insertion
            memmove(&m_pString[position1 + amount], &m_pString[position1], current - position1);
        }
        // Insert data into the current string
        memcpy(&m_pString[position1], &sString.m_pString[position2], amount);

        // Terminate the string
        m_pString[current + amount] = '\0';
    }
    return *this;

} // insert

//******************************************************************************

CString&
CString::insert
(
    size_t              position1,
    const char         *pString,
    size_t              position2,
    size_t              length
)
{
    size_t              current;
    size_t              space;
    size_t              amount = 0;

    // Check for an invalid position
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    // Compute the amount of space required
    if (pString != NULL)
    {
        amount = strnlen(&pString[position2], length);
    }
    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to insert in the current string
    if (amount != 0)
    {
        // Check for data to move before insertion
        if (position1 != size())
        {
            // Move current string data to make space for insertion
            memmove(&m_pString[position1 + amount], &m_pString[position1], current - position1);
        }
        // Insert data into the current string
        memcpy(&m_pString[position1], &pString[position2], amount);

        // Terminate the string
        m_pString[current + amount] = '\0';
    }
    return *this;

} // insert

//******************************************************************************

CString&
CString::insert
(
    size_t              position1,
    size_t              length,
    char                c
)
{
    size_t              current;
    size_t              space;

    // Check for an invalid position
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + length) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check for data to insert in the current string
    if (length != 0)
    {
        // Check for data to move before insertion
        if (position1 != size())
        {
            // Move current string data to make space for insertion
            memmove(&m_pString[position1 + length], &m_pString[position1], current - position1);
        }
        // Insert given character into the current string
        memset(&m_pString[position1], c, length);

        // Terminate the string
        m_pString[current + length] = '\0';
    }
    return *this;

} // insert

//******************************************************************************

size_t
CString::length() const
{
    size_t              length = 0;

    // Get the string length (Don't exceed allocated capacity)
    length = strnlen(m_pString, capacity() - 1);

    return length;

} // length

//******************************************************************************

CString&
CString::lower()
{
    // Colwert string to lowercase
    _strlwr(m_pString);

    return *this;

} // lower

//******************************************************************************

CString&
CString::replace
(
    size_t              position1,
    size_t              length1,
    const CString&      sString,
    size_t              position2,
    size_t              length2
)
{
    size_t              current;
    size_t              space;
    size_t              length  = 0;
    size_t              replace = 0;
    size_t              amount  = 0;

    // Check for invalid positions
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    if (position2 > sString.size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position2, sString.size());
    }
    // Compute the length of string to replace
    length = strnlen(&m_pString[position1], min(length1, (capacity() - position1)));

    // Compute the length of replacement string
    replace = strnlen(&sString.m_pString[position2], min(length2, (sString.capacity() - position2)));

    // Compute the additional amount of space needed
    if (replace > length)
    {
        amount = replace - length;
    }
    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check to see if current string needs to be moved
    if (length != replace)
    {
        // Check for a longer replacement string
        if (length < replace)
        {
            // Move string right to make room for larger replacement string
            memmove(&m_pString[position1 + (replace - length)], &m_pString[position1], (current - position1));
        }
        else    // Shorter replacement string
        {
            // Move string left to since replacement string is shorter
            memmove(&m_pString[position1 - (length - replace)], &m_pString[position1], (current - position1));
        }
    }
    // Replace current string with the replacement string
    memcpy(&m_pString[position1], &sString.m_pString[position2], replace);

    // Terminate the string
    if (length < replace)
    {
        m_pString[current + (replace - length)] = '\0';
    }
    else
    {
        m_pString[current + (length - replace)] = '\0';
    }
    return *this;

} // replace

//******************************************************************************

CString&
CString::replace
(
    size_t              position1,
    size_t              length1,
    const char         *pString,
    size_t              position2,
    size_t              length2
)
{
    size_t              current;
    size_t              space;
    size_t              length  = 0;
    size_t              replace = 0;
    size_t              amount  = 0;

    // Check for an invalid position
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    // Compute the length of string to replace
    length = strnlen(&m_pString[position1], min(length1, (capacity() - position1)));

    // Compute the length of replacement string
    if (pString != NULL)
    {
        replace = strnlen(&pString[position2], length2);
    }
    // Compute the additional amount of space needed
    if (replace > length)
    {
        amount = replace - length;
    }
    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check to see if current string needs to be moved
    if (length != replace)
    {
        // Check for a longer replacement string
        if (length < replace)
        {
            // Move string right to make room for larger replacement string
            memmove(&m_pString[position1 + (replace - length)], &m_pString[position1], (current - position1));
        }
        else    // Shorter replacement string
        {
            // Move string left to since replacement string is shorter
            memmove(&m_pString[position1 - (length - replace)], &m_pString[position1], (current - position1));
        }
    }
    // Check for string replacement and termination needed
    if (pString != NULL)
    {
        // Replace current string with the replacement string
        memcpy(&m_pString[position1], &pString[position2], replace);

        // Terminate the string
        if (length < replace)
        {
            m_pString[current + (replace - length)] = '\0';
        }
        else
        {
            m_pString[current + (length - replace)] = '\0';
        }
    }
    return *this;

} // replace

//******************************************************************************

CString&
CString::replace
(
    size_t              position1,
    size_t              length1,
    size_t              length2,
    char                c
)
{
    size_t              current;
    size_t              space;
    size_t              length  = 0;
    size_t              amount  = 0;

    // Check for an invalid position
    if (position1 > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position1, size());
    }
    // Compute the length of string to replace
    length = strnlen(&m_pString[position1], min(length1, (capacity() - position1)));

    // Compute the additional amount of space needed
    if (length2 > length)
    {
        amount = length2 - length;
    }
    // Get the current string size [length]
    current = size();

    // Compute amount of space that needs to be allocated
    space = capacity();
    while ((current + amount) >= space)
    {
        space = (space << 1) + 1;
    }
    // Check to see if new space needs to be allocated
    if (space != capacity())
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, space));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(space, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Check to see if current string needs to be moved
    if (length != length2)
    {
        // Check for a longer replacement string
        if (length < length2)
        {
            // Move string right to make room for larger replacement string
            memmove(&m_pString[position1 + (length2 - length)], &m_pString[position1], (current - position1));
        }
        else    // Shorter replacement string
        {
            // Move string left to since replacement string is shorter
            memmove(&m_pString[position1 - (length - length2)], &m_pString[position1], (current - position1));
        }
    }
    // Fill current string with the replacement character
    memset(&m_pString[position1], c, length2);

    // Terminate the string
    if (length < length2)
    {
        m_pString[current + (length2 - length)] = '\0';
    }
    else
    {
        m_pString[current + (length - length2)] = '\0';
    }
    return *this;

} // replace

//******************************************************************************

CString&
CString::reserve
(
    size_t              size
)
{
    size_t              current;

    // Get the current string length
    current = length();

    // Make sure reserve size won't truncate the current string
    if (size > current)
    {
        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, size));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(size, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    return *this;

} // reserve

//******************************************************************************

CString&
CString::resize
(
    size_t              length,
    char                c
)
{
    size_t              current;
    size_t              newSize;

    // Get the current string size (length)
    current = size();

    // Check for not clearing the current string (resize to 0)
    if (length != 0)
    {
        // Compute the new allocation size
        newSize = length + 1;

        // Try to reallocate the required space
        m_pString = charptr(stringRealloc(m_pString, newSize));
        if (m_pString == NULL)
        {
            // Throw error indicating unable to reallocate string
            throw CMemoryException(newSize, __FILE__, __FUNCTION__, __LINE__);
        }
        // Check for fill characters required
        if (length > current)
        {
            // Fill new characters with the given character
            memset(&m_pString[current], c, (length - current));

            // Terminate the string (if necessary)
            if (c != 0)
            {
                m_pString[length] = '\0';
            }
        }
        else    // No fill characters required
        {
            // Terminate the string
            m_pString[capacity() - 1] = '\0';
        }
    }
    else    // Resize to zero
    {
        // Free existing string allocation
        stringFree(m_pString);

        // Allocate enough space for a NULL string
        m_pString = charptr(stringAlloc(1));
        if (m_pString != NULL)
        {
            // NULL terminate the string
            m_pString[0] = '\0';
        }
        else    // String allocation error
        {
            // Throw memory error
            throw CMemoryException(1, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    return *this;

} // resize

//******************************************************************************

size_t
CString::size() const
{
    size_t              size = 0;

    // Get the string size (Don't exceed allocated capacity)
    size = strnlen(m_pString, capacity() - 1);

    return size;

} // size

//******************************************************************************

int
CString::sprintf
(
    const char         *pszFormat,
    ...
)
{
    va_list             va;
    int                 nLength;

    assert(pszFormat != NULL);

    // Check for a NULL string (if so need to reserve space)
    if (capacity() == 1)
    {
        // Reserve the default sprintf string size
        reserve(STRING_SPRINTF_SIZE);
    }
    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Perform the printf into the string buffer
    nLength = _vsnprintf(m_pString, capacity(), pszFormat, va);

    // Check for output truncated
    if (nLength == -1)
    {
        // Set length to string size
        nLength = static_cast<int>(capacity());
    }
    return nLength;

} // sprintf

//******************************************************************************

const char*
CString::str
(
    size_t              position
) const
{
    // Check for an invalid position
    if (position > length())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, length());
    }
    // Make sure the string is terminated
    m_pString[length()] = '\0';

    return &m_pString[position];

} // str

//******************************************************************************

CString
CString::substr
(
    size_t              position,
    size_t              length
)
{
    CString             string;

    // Check for an invalid position
    if (position > size())
    {
        throw CStringException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid string position (%d > %d)",
                               position, size());
    }
    // Return the desired substring
    return string.assign(*this, position, length);

} // substr

//******************************************************************************

void
CString::swap
(
    CString&            sString
)
{
    char               *pString;

    // Swap the strings
    pString = m_pString;

    m_pString = sString.m_pString;

    sString.m_pString = pString;

} // swap

//******************************************************************************

CString&
CString::upper()
{
    // Colwert string to uppercase
    _strupr(m_pString);

    return *this;

} // upper

//******************************************************************************

const CString&
CString::operator=
(
    const char         *pString
)
{
    size_t              length;

    // Check for string given
    if (pString != NULL)
    {
        // Get the length of the given string (plus room for terminator)
        length = strlen(pString) + 1;

        // Check to see if the given string will need a new allocation
        if (length >= capacity())
        {
            // Free the current allocation
            stringFree(m_pString);

            // Try to allocate the required space
            m_pString = charptr(stringAlloc(length));
            if (m_pString == NULL)
            {
                // Throw memory error
                throw CMemoryException(length, __FILE__, __FUNCTION__, __LINE__);
            }
        }
        // Copy the given string to the new string buffer
        strcpy(m_pString, pString);
    }
    else    // NULL assignment
    {
        // Set string to the NULL string
        m_pString[0] = 0;
    }
    return *this;

} // operator=

//******************************************************************************

const CString&
CString::operator=
(
    const CString&      sString
)
{
    size_t              length;

    // Get the length of the given string (plus room for terminator)
    length = strnlen(sString.m_pString, sString.capacity()) + 1;

    // Check to see if the given string will need a new allocation
    if (length >= capacity())
    {
        // Free the current allocation
        stringFree(m_pString);

        // Try to allocate the required space
        m_pString = charptr(stringAlloc(length));
        if (m_pString == NULL)
        {
            // Throw memory error
            throw CMemoryException(length, __FILE__, __FUNCTION__, __LINE__);
        }
    }
    // Copy the given string to the new string buffer
    strcpy(m_pString, sString.m_pString);

    return *this;

} // operator=

//******************************************************************************

CString
CString::operator+
(
    const char         *pString
) const
{
    size_t              amount = length() + 1;
    CString             sNewString;

    // Check for string given
    if (pString != NULL)
    {
        amount += strlen(pString);
    }
    // Create a new copy of the original string (Plus room for new string)
    sNewString = CString(*this, 0, amount);
    if (sNewString == NULL)
    {
        // Throw error indicating unable to allocate string
        throw CMemoryException(amount, __FILE__, __FUNCTION__, __LINE__);
    }
    // Append given string to the new string
    sNewString.append(pString);

    return sNewString;

} // operator+

//******************************************************************************

CString
CString::operator+
(
    const CString&      sString
) const
{
    size_t              amount = length() + 1;
    CString             sNewString;

    // Adjust amount to allocate to include new string
    amount += sString.length();

    // Create a new copy of the original string (Plus room for new string)
    sNewString = CString(*this, 0, amount);
    if (sNewString == NULL)
    {
        // Throw error indicating unable to allocate string
        throw CMemoryException(amount, __FILE__, __FUNCTION__, __LINE__);
    }
    // Append given string to the new string
    sNewString.append(sString);

    return sNewString;

} // operator+

//******************************************************************************

const CString&
CString::operator+=
(
    const char         *pString
)
{
    size_t              current = length();
    size_t              length;
    size_t              size;

    // Check for string given
    if (pString != NULL)
    {
        // Get length of the given string (May be zero)
        length = strlen(pString);
        if (length != 0)
        {
            // Check for reallocation required
            if ((current + length) >= capacity())
            {
                // Compute the new allocated size
                size = current + length + 1;

                // Try to reallocate the required space
                m_pString = charptr(stringRealloc(m_pString, size));
                if (m_pString == NULL)
                {
                    // Throw error indicating unable to reallocate string
                    throw CMemoryException(size, __FILE__, __FUNCTION__, __LINE__);
                }
            }
            // Copy new string to end of current string
            strcpy(&m_pString[current], pString);
        }
    }
    return *this;

} // operator+=

//******************************************************************************

const CString&
CString::operator+=
(
    const CString&      sString
)
{
    size_t              current = length();
    size_t              length;
    size_t              size;

    // Get length of the given string
    length = strnlen(sString.m_pString, sString.capacity());
    if (length != 0)
    {
        // Check for reallocation required
        if ((current + length) >= capacity())
        {
            // Compute the new allocated size
            size = current + length + 1;

            // Try to reallocate the required space
            m_pString = charptr(stringRealloc(m_pString, size));
            if (m_pString == NULL)
            {
                // Throw error indicating unable to reallocate string
                throw CMemoryException(size, __FILE__, __FUNCTION__, __LINE__);
            }
        }
        // Copy new string to end of current string
        strcpy(&m_pString[current], sString);
    }
    return *this;

} // operator+=

//******************************************************************************

static void*
stringAlloc
(
    size_t              size
)
{
    STRING_DATA        *pStringData;
    void               *pString = NULL;

    // Try to allocate the requested space (Includes string data)
    pStringData = static_cast<STRING_DATA*>(malloc(size + sizeof(STRING_DATA)));
    if (pStringData != NULL)
    {
        // Save the allocated size
        pStringData->allocated = size;

        // Adjust return pointer to not include string data
        pString = reinterpret_cast<char*>(pStringData) + sizeof(STRING_DATA);
    }
    return pString;

} // stringAlloc

//******************************************************************************

static void*
stringRealloc
(
    void               *pString,
    size_t              size
)
{
    STRING_DATA        *pStringData;

    assert(pString != NULL);

    // Compute string data pointer from string pointer
    pStringData = reinterpret_cast<STRING_DATA*>(static_cast<char*>(pString) - sizeof(STRING_DATA));

    // Try to reallocate to the new requested size (Includes string data)
    pStringData = reinterpret_cast<STRING_DATA*>(realloc(pStringData, (size + sizeof(STRING_DATA))));
    if (pStringData != NULL)
    {
        // Update the allocated string size
        pStringData->allocated = size;

        // Adjust return pointer to not include string data
        pString = reinterpret_cast<char*>(pStringData) + sizeof(STRING_DATA);
    }
    else    // Failed to reallocate string
    {
        // Indicate reallocation failure
        pString = NULL;
    }
    return pString;

} // stringRealloc

//******************************************************************************

static void
stringFree
(
    void               *pString
)
{
    STRING_DATA        *pStringData;

    assert(pString != NULL);

    // Compute string data pointer from given string
    pStringData = reinterpret_cast<STRING_DATA*>(static_cast<char*>(pString) - sizeof(STRING_DATA));

    // Free the string data (and string)
    free(pStringData);

} // stringFree

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
