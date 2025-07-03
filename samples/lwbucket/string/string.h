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
|*  Module: string.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _STRING_H
#define _STRING_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Define the not found location value for string find
#define NOT_FOUND                       (static_cast<size_t>(-1))
#define STRING_SPRINTF_SIZE             4096    // Default string sprintf size

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _STRING_DATA
{
    size_t              allocated;

} STRING_DATA, *PSTRING_DATA;

//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define STR(string)             ((string).str())
#define VARSTR(width, string)   (width), STR(string)

//******************************************************************************
//
// class CString
//
//******************************************************************************
class CString
{
private:
mutable char*           m_pString;              // Must be the only data member in this class

        STRING_DATA*    stringData();
public:
                        CString();
                        CString(const CString& sString, size_t position = 0, size_t length = -1);
                        CString(const char* pString, size_t length = -1);
                        CString(size_t length, char c = 0);
                       ~CString();

const   CString&        operator=(const char* pString);
const   CString&        operator=(const CString& sString);
        CString         operator+(const char* pString) const;
        CString         operator+(const CString& sString) const;
const   CString&        operator+=(const char* pString);
const   CString&        operator+=(const CString& sString);
        char&           operator[](int element)         { return m_pString[element]; }
                        operator const char*() const    { return m_pString; }

        CString&        append(const CString& sString, size_t position = 0, size_t length = -1);
        CString&        append(const char* pString, size_t position = 0, size_t length = -1);
        CString&        append(size_t length, char c = 0);
        CString&        assign(const CString& sString, size_t position = 0, size_t length = -1);
        CString&        assign(const char* pString, size_t position = 0, size_t length = -1);
        CString&        assign(size_t length, char c = 0);
        char            at(size_t position = 0) const;
        size_t          capacity() const;
        void            clear();
        int             compare(const CString& string) const;
        int             compare(const char* pString) const;
        int             compare(size_t position1, size_t length1, const CString& string, size_t position2 = 0, size_t length2 = -1) const;
        int             compare(size_t position1, size_t length1, const char* pString, size_t position2 = 0, size_t length2 = -1) const;
        size_t          copy(char* pString, size_t length = -1, size_t position = 0);
        char*           data(size_t position = 0);
        bool            empty() const                   { return (length() == 0); }
        CString&        erase(size_t position = 0, size_t length = -1);
        CString&        fill(char c, size_t length = 1);
        CString&        fill(size_t position, size_t length, char c);
        size_t          find(const CString& string, size_t position = 0, size_t length = -1) const;
        size_t          find(const char* pString, size_t position = 0, size_t length = -1) const;
        size_t          find(char c, size_t position = 0, size_t length = -1) const;
        CString&        insert(size_t position1, const CString& string, size_t position2 = 0, size_t length = -1);
        CString&        insert(size_t position1, const char* pString, size_t position2 = 0, size_t length = -1);
        CString&        insert(size_t position1, size_t length, char c);
        size_t          length() const;
        CString&        lower();
        bool            null() const                    { return (m_pString == NULL); }
        CString&        replace(size_t position1, size_t length1, const CString& string, size_t position2 = 0, size_t length2 = -1);
        CString&        replace(size_t position1, size_t length1, const char* pString, size_t position2 = 0, size_t length2 = -1);
        CString&        replace(size_t position1, size_t length1, size_t length2, char c);
        CString&        reserve(size_t size = 0);
        CString&        resize(size_t length = 0, char c = 0);
        size_t          size() const;
        int             sprintf(const char *pszFormat, ...);
        const char*     str(size_t position = 0) const;
        CString         substr(size_t position = 0, size_t length = -1);
        void            swap(CString& string);
        CString&        upper();

}; // class CString

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _STRING_H
