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
|*  Module: pointer.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _POINTER_H
#define _POINTER_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Address types
enum AddressType
{
    CpuVirtual = 0,                         // CPU virtual address
    CpuPhysical,                            // CPU physical address

}; // AddressType

// Pointer data types
enum PointerType
{
    ThreadPointer = 0,                      // Thread pointer data type
    ProcessPointer,                         // Process pointer data type
    SessionPointer,                         // Session pointer data type

}; // PointerType

//******************************************************************************
//
// Forwards (These functions are required by the class/template definitions)
//
//******************************************************************************
extern  ULONG                       pointerSize();
extern  ULONG64                     pointerMask();
extern  ULONG                       pointerWidth();

//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define EXTEND(Pointer)             (static_cast<LONG64>(static_cast<LONG>(Pointer)))
#define TARGET(Pointer)             ((Pointer) & pointerMask())
#define UTOP(Pointer)               pointerWidth(), ((Pointer) & pointerMask())
#define ADDR(Address)               pointerWidth(), (Address).addr()
#define PTR(Pointer)                pointerWidth(), (Pointer).ptr()
#define NULL_PTR()                  pointerWidth(), 0ull

//******************************************************************************
//
//  Template TARGET_ADDR (Target system address template)
//
//******************************************************************************
template <AddressType T>
class TARGET_ADDR
{
private:
static  const   AddressType m_Type = T;

public:
        ULONG64         m_ulAddress;

public:
                        TARGET_ADDR<T>()                    { m_ulAddress = 0; }
                        TARGET_ADDR<T>(ULONG64 ulAddress)   { m_ulAddress = ulAddress; }

        ULONG64         address() const                     { return ((m_ulAddress < 0x100000000) ? EXTEND(m_ulAddress) : m_ulAddress); }
        ULONG64         addr() const                        { return m_ulAddress; }

        PULONG64        data()                              { return &m_ulAddress; }

        AddressType     type() const                        { return m_Type; }

// Compound assignment operator overloads
const   TARGET_ADDR<T>& operator+=(const CHAR signedValue)
                        {
                            m_ulAddress += signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const UCHAR unsignedValue)
                        {
                            m_ulAddress += unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const SHORT signedValue)
                        {
                            m_ulAddress += signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const USHORT unsignedValue)
                        {
                            m_ulAddress += unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const INT signedValue)
                        {
                            m_ulAddress += signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const UINT unsignedValue)
                        {
                            m_ulAddress += unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const LONG signedValue)
                        {
                            m_ulAddress += signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const ULONG unsignedValue)
                        {
                            m_ulAddress += unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const LONG64 signedValue)
                        {
                            m_ulAddress += signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator+=(const ULONG64 unsignedValue)
                        {
                            m_ulAddress += unsignedValue;

                            return (*this);
                        }

const   TARGET_ADDR<T>& operator-=(const CHAR signedValue)
                        {
                            m_ulAddress -= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const UCHAR unsignedValue)
                        {
                            m_ulAddress -= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const SHORT signedValue)
                        {
                            m_ulAddress -= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const USHORT unsignedValue)
                        {
                            m_ulAddress -= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const INT signedValue)
                        {
                            m_ulAddress -= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const UINT unsignedValue)
                        {
                            m_ulAddress -= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const LONG signedValue)
                        {
                            m_ulAddress -= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const ULONG unsignedValue)
                        {
                            m_ulAddress -= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const LONG64 signedValue)
                        {
                            m_ulAddress -= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator-=(const ULONG64 unsignedValue)
                        {
                            m_ulAddress -= unsignedValue;

                            return (*this);
                        }

const   TARGET_ADDR<T>& operator&=(const CHAR signedValue)
                        {
                            m_ulAddress &= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const UCHAR unsignedValue)
                        {
                            m_ulAddress &= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const SHORT signedValue)
                        {
                            m_ulAddress &= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const USHORT unsignedValue)
                        {
                            m_ulAddress &= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const INT signedValue)
                        {
                            m_ulAddress &= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const UINT unsignedValue)
                        {
                            m_ulAddress &= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const LONG signedValue)
                        {
                            m_ulAddress &= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const ULONG unsignedValue)
                        {
                            m_ulAddress &= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const LONG64 signedValue)
                        {
                            m_ulAddress &= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator&=(const ULONG64 unsignedValue)
                        {
                            m_ulAddress &= unsignedValue;

                            return (*this);
                        }

const   TARGET_ADDR<T>& operator|=(const CHAR signedValue)
                        {
                            m_ulAddress |= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const UCHAR unsignedValue)
                        {
                            m_ulAddress |= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const SHORT signedValue)
                        {
                            m_ulAddress |= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const USHORT unsignedValue)
                        {
                            m_ulAddress |= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const INT signedValue)
                        {
                            m_ulAddress |= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const UINT unsignedValue)
                        {
                            m_ulAddress |= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const LONG signedValue)
                        {
                            m_ulAddress |= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const ULONG unsignedValue)
                        {
                            m_ulAddress |= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const LONG64 signedValue)
                        {
                            m_ulAddress |= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator|=(const ULONG64 unsignedValue)
                        {
                            m_ulAddress |= unsignedValue;

                            return (*this);
                        }

const   TARGET_ADDR<T>& operator^=(const CHAR signedValue)
                        {
                            m_ulAddress ^= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const UCHAR unsignedValue)
                        {
                            m_ulAddress ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const SHORT signedValue)
                        {
                            m_ulAddress ^= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const USHORT unsignedValue)
                        {
                            m_ulAddress ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const INT signedValue)
                        {
                            m_ulAddress ^= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const UINT unsignedValue)
                        {
                            m_ulAddress ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const LONG signedValue)
                        {
                            m_ulAddress ^= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const ULONG unsignedValue)
                        {
                            m_ulAddress ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const LONG64 signedValue)
                        {
                            m_ulAddress ^= signedValue;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator^=(const ULONG64 unsignedValue)
                        {
                            m_ulAddress ^= unsignedValue;

                            return (*this);
                        }
// Unary operator overloads
const   TARGET_ADDR<T>& operator++()
                        {
                            ++m_ulAddress;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator++(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ulAddress++;

                            return (*this);
                        }

const   TARGET_ADDR<T>& operator--()
                        {
                            --m_ulAddress;

                            return (*this);
                        }
const   TARGET_ADDR<T>& operator--(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ulAddress--;

                            return (*this);
                        }
        // Binary operator overloads
        TARGET_ADDR<T>  operator+(const CHAR signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += signedValue);
                        }
        TARGET_ADDR<T>  operator+(const UCHAR unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += unsignedValue);
                        }
        TARGET_ADDR<T>  operator+(const SHORT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += signedValue);
                        }
        TARGET_ADDR<T>  operator+(const USHORT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += unsignedValue);
                        }
        TARGET_ADDR<T>  operator+(const INT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += signedValue);
                        }
        TARGET_ADDR<T>  operator+(const UINT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += unsignedValue);
                        }
        TARGET_ADDR<T>  operator+(const LONG signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += signedValue);
                        }
        TARGET_ADDR<T>  operator+(const ULONG unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += unsignedValue);
                        }
        TARGET_ADDR<T>  operator+(const LONG64 signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += signedValue);
                        }
        TARGET_ADDR<T>  operator+(const ULONG64 unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) += unsignedValue);
                        }

        TARGET_ADDR<T>  operator-(const CHAR signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= signedValue);
                        }
        TARGET_ADDR<T>  operator-(const UCHAR unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= unsignedValue);
                        }
        TARGET_ADDR<T>  operator-(const SHORT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= signedValue);
                        }
        TARGET_ADDR<T>  operator-(const USHORT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= unsignedValue);
                        }
        TARGET_ADDR<T>  operator-(const INT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= signedValue);
                        }
        TARGET_ADDR<T>  operator-(const UINT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= unsignedValue);
                        }
        TARGET_ADDR<T>  operator-(const LONG signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= signedValue);
                        }
        TARGET_ADDR<T>  operator-(const ULONG unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= unsignedValue);
                        }
        TARGET_ADDR<T>  operator-(const LONG64 signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= signedValue);
                        }
        TARGET_ADDR<T>  operator-(const ULONG64 unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) -= unsignedValue);
                        }
        ULONG64         operator-(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress - targetAddress.m_ulAddress);
                        }

        ULONG64         operator*(const CHAR signedValue) const
                        {
                            return (m_ulAddress * signedValue);
                        }
        ULONG64         operator*(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress * unsignedValue);
                        }
        ULONG64         operator*(const SHORT signedValue) const
                        {
                            return (m_ulAddress * signedValue);
                        }
        ULONG64         operator*(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress * unsignedValue);
                        }
        ULONG64         operator*(const INT signedValue) const
                        {
                            return (m_ulAddress * signedValue);
                        }
        ULONG64         operator*(const UINT unsignedValue) const
                        {
                            return (m_ulAddress * unsignedValue);
                        }
        ULONG64         operator*(const LONG signedValue) const
                        {
                            return (m_ulAddress * signedValue);
                        }
        ULONG64         operator*(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress * unsignedValue);
                        }
        ULONG64         operator*(const LONG64 signedValue) const
                        {
                            return (m_ulAddress * signedValue);
                        }
        ULONG64         operator*(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress * unsignedValue);
                        }

        ULONG64         operator/(const CHAR signedValue) const
                        {
                            return (m_ulAddress / signedValue);
                        }
        ULONG64         operator/(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress / unsignedValue);
                        }
        ULONG64         operator/(const SHORT signedValue) const
                        {
                            return (m_ulAddress / signedValue);
                        }
        ULONG64         operator/(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress / unsignedValue);
                        }
        ULONG64         operator/(const INT signedValue) const
                        {
                            return (m_ulAddress / signedValue);
                        }
        ULONG64         operator/(const UINT unsignedValue) const
                        {
                            return (m_ulAddress / unsignedValue);
                        }
        ULONG64         operator/(const LONG signedValue) const
                        {
                            return (m_ulAddress / signedValue);
                        }
        ULONG64         operator/(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress / unsignedValue);
                        }
        ULONG64         operator/(const LONG64 signedValue) const
                        {
                            return (m_ulAddress / signedValue);
                        }
        ULONG64         operator/(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress / unsignedValue);
                        }

        ULONG64         operator%(const CHAR signedValue) const
                        {
                            return (m_ulAddress % signedValue);
                        }
        ULONG64         operator%(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress % unsignedValue);
                        }
        ULONG64         operator%(const SHORT signedValue) const
                        {
                            return (m_ulAddress % signedValue);
                        }
        ULONG64         operator%(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress % unsignedValue);
                        }
        ULONG64         operator%(const INT signedValue) const
                        {
                            return (m_ulAddress % signedValue);
                        }
        ULONG64         operator%(const UINT unsignedValue) const
                        {
                            return (m_ulAddress % unsignedValue);
                        }
        ULONG64         operator%(const LONG signedValue) const
                        {
                            return (m_ulAddress % signedValue);
                        }
        ULONG64         operator%(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress % unsignedValue);
                        }
        ULONG64         operator%(const LONG64 signedValue) const
                        {
                            return (m_ulAddress % signedValue);
                        }
        ULONG64         operator%(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress % unsignedValue);
                        }
        // Bitwise operator overloads
        TARGET_ADDR<T>  operator~()
                        {
                            m_ulAddress = ~m_ulAddress;

                            return (*this);
                        }

        TARGET_ADDR<T>  operator&(const CHAR signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= signedValue);
                        }
        TARGET_ADDR<T>  operator&(const UCHAR unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= unsignedValue);
                        }
        TARGET_ADDR<T>  operator&(const SHORT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= signedValue);
                        }
        TARGET_ADDR<T>  operator&(const USHORT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= unsignedValue);
                        }
        TARGET_ADDR<T>  operator&(const INT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= signedValue);
                        }
        TARGET_ADDR<T>  operator&(const UINT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= unsignedValue);
                        }
        TARGET_ADDR<T>  operator&(const LONG signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= signedValue);
                        }
        TARGET_ADDR<T>  operator&(const ULONG unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= unsignedValue);
                        }
        TARGET_ADDR<T>  operator&(const LONG64 signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= signedValue);
                        }
        TARGET_ADDR<T>  operator&(const ULONG64 unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) &= unsignedValue);
                        }

        TARGET_ADDR<T>  operator|(const CHAR signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= signedValue);
                        }
        TARGET_ADDR<T>  operator|(const UCHAR unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= unsignedValue);
                        }
        TARGET_ADDR<T>  operator|(const SHORT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= signedValue);
                        }
        TARGET_ADDR<T>  operator|(const USHORT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= unsignedValue);
                        }
        TARGET_ADDR<T>  operator|(const INT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= signedValue);
                        }
        TARGET_ADDR<T>  operator|(const UINT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= unsignedValue);
                        }
        TARGET_ADDR<T>  operator|(const LONG signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= signedValue);
                        }
        TARGET_ADDR<T>  operator|(const ULONG unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= unsignedValue);
                        }
        TARGET_ADDR<T>  operator|(const LONG64 signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= signedValue);
                        }
        TARGET_ADDR<T>  operator|(const ULONG64 unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) |= unsignedValue);
                        }

        TARGET_ADDR<T>  operator^(const CHAR signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= signedValue);
                        }
        TARGET_ADDR<T>  operator^(const UCHAR unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= unsignedValue);
                        }
        TARGET_ADDR<T>  operator^(const SHORT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= signedValue);
                        }
        TARGET_ADDR<T>  operator^(const USHORT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= unsignedValue);
                        }
        TARGET_ADDR<T>  operator^(const INT signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= signedValue);
                        }
        TARGET_ADDR<T>  operator^(const UINT unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= unsignedValue);
                        }
        TARGET_ADDR<T>  operator^(const LONG signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= signedValue);
                        }
        TARGET_ADDR<T>  operator^(const ULONG unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= unsignedValue);
                        }
        TARGET_ADDR<T>  operator^(const LONG64 signedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= signedValue);
                        }
        TARGET_ADDR<T>  operator^(const ULONG64 unsignedValue) const
                        {
                            return (TARGET_ADDR<T>(*this) ^= unsignedValue);
                        }

        ULONG64         operator>>(const CHAR signedValue) const
                        {
                            return (m_ulAddress >> signedValue);
                        }
        ULONG64         operator>>(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress >> unsignedValue);
                        }
        ULONG64         operator>>(const SHORT signedValue) const
                        {
                            return (m_ulAddress >> signedValue);
                        }
        ULONG64         operator>>(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress >> unsignedValue);
                        }
        ULONG64         operator>>(const INT signedValue) const
                        {
                            return (m_ulAddress >> signedValue);
                        }
        ULONG64         operator>>(const UINT unsignedValue) const
                        {
                            return (m_ulAddress >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG signedValue) const
                        {
                            return (m_ulAddress >> signedValue);
                        }
        ULONG64         operator>>(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG64 signedValue) const
                        {
                            return (m_ulAddress >> signedValue);
                        }
        ULONG64         operator>>(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress >> unsignedValue);
                        }

        ULONG64         operator<<(const CHAR signedValue) const
                        {
                            return (m_ulAddress << signedValue);
                        }
        ULONG64         operator<<(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress << unsignedValue);
                        }
        ULONG64         operator<<(const SHORT signedValue) const
                        {
                            return (m_ulAddress << signedValue);
                        }
        ULONG64         operator<<(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress << unsignedValue);
                        }
        ULONG64         operator<<(const INT signedValue) const
                        {
                            return (m_ulAddress << signedValue);
                        }
        ULONG64         operator<<(const UINT unsignedValue) const
                        {
                            return (m_ulAddress << unsignedValue);
                        }
        ULONG64         operator<<(const LONG signedValue) const
                        {
                            return (m_ulAddress << signedValue);
                        }
        ULONG64         operator<<(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress << unsignedValue);
                        }
        ULONG64         operator<<(const LONG64 signedValue) const
                        {
                            return (m_ulAddress << signedValue);
                        }
        ULONG64         operator<<(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress << unsignedValue);
                        }
        // Logical operator overloads
        bool            operator==(const CHAR signedValue) const
                        {
                            return (m_ulAddress == static_cast<UCHAR>(signedValue));
                        }
        bool            operator==(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress == unsignedValue);
                        }
        bool            operator==(const SHORT signedValue) const
                        {
                            return (m_ulAddress == static_cast<USHORT>(signedValue));
                        }
        bool            operator==(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress == unsignedValue);
                        }
        bool            operator==(const INT signedValue) const
                        {
                            return (m_ulAddress == static_cast<UINT>(signedValue));
                        }
        bool            operator==(const UINT unsignedValue) const
                        {
                            return (m_ulAddress == unsignedValue);
                        }
        bool            operator==(const LONG signedValue) const
                        {
                            return (m_ulAddress == static_cast<ULONG>(signedValue));
                        }
        bool            operator==(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress == unsignedValue);
                        }
        bool            operator==(const LONG64 signedValue) const
                        {
                            return (m_ulAddress == static_cast<ULONG64>(signedValue));
                        }
        bool            operator==(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress == unsignedValue);
                        }
        bool            operator==(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress == targetAddress.addr());
                        }

        bool            operator!=(const CHAR signedValue) const
                        {
                            return (m_ulAddress != static_cast<UCHAR>(signedValue));
                        }
        bool            operator!=(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress != unsignedValue);
                        }
        bool            operator!=(const SHORT signedValue) const
                        {
                            return (m_ulAddress != static_cast<USHORT>(signedValue));
                        }
        bool            operator!=(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress != unsignedValue);
                        }
        bool            operator!=(const INT signedValue) const
                        {
                            return (m_ulAddress != static_cast<UINT>(signedValue));
                        }
        bool            operator!=(const UINT unsignedValue) const
                        {
                            return (m_ulAddress != unsignedValue);
                        }
        bool            operator!=(const LONG signedValue) const
                        {
                            return (m_ulAddress != static_cast<ULONG>(signedValue));
                        }
        bool            operator!=(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress != unsignedValue);
                        }
        bool            operator!=(const LONG64 signedValue) const
                        {
                            return (m_ulAddress != static_cast<ULONG64>(signedValue));
                        }
        bool            operator!=(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress != unsignedValue);
                        }
        bool            operator!=(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress != targetAddress.addr());
                        }

        bool            operator>(const CHAR signedValue) const
                        {
                            return (m_ulAddress > static_cast<UCHAR>(signedValue));
                        }
        bool            operator>(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress > unsignedValue);
                        }
        bool            operator>(const SHORT signedValue) const
                        {
                            return (m_ulAddress > static_cast<USHORT>(signedValue));
                        }
        bool            operator>(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress > unsignedValue);
                        }
        bool            operator>(const INT signedValue) const
                        {
                            return (m_ulAddress > static_cast<UINT>(signedValue));
                        }
        bool            operator>(const UINT unsignedValue) const
                        {
                            return (m_ulAddress > unsignedValue);
                        }
        bool            operator>(const LONG signedValue) const
                        {
                            return (m_ulAddress > static_cast<ULONG>(signedValue));
                        }
        bool            operator>(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress > unsignedValue);
                        }
        bool            operator>(const LONG64 signedValue) const
                        {
                            return (m_ulAddress > static_cast<ULONG64>(signedValue));
                        }
        bool            operator>(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress > unsignedValue);
                        }
        bool            operator>(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress > targetAddress.addr());
                        }

        bool            operator<(const CHAR signedValue) const
                        {
                            return (m_ulAddress < static_cast<UCHAR>(signedValue));
                        }
        bool            operator<(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress < unsignedValue);
                        }
        bool            operator<(const SHORT signedValue) const
                        {
                            return (m_ulAddress < static_cast<USHORT>(signedValue));
                        }
        bool            operator<(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress < unsignedValue);
                        }
        bool            operator<(const INT signedValue) const
                        {
                            return (m_ulAddress < static_cast<UINT>(signedValue));
                        }
        bool            operator<(const UINT unsignedValue) const
                        {
                            return (m_ulAddress < unsignedValue);
                        }
        bool            operator<(const LONG signedValue) const
                        {
                            return (m_ulAddress < static_cast<ULONG>(signedValue));
                        }
        bool            operator<(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress < unsignedValue);
                        }
        bool            operator<(const LONG64 signedValue) const
                        {
                            return (m_ulAddress < static_cast<ULONG64>(signedValue));
                        }
        bool            operator<(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress < unsignedValue);
                        }
        bool            operator<(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress < targetAddress.addr());
                        }

        bool            operator<=(const CHAR signedValue) const
                        {
                            return (m_ulAddress <= static_cast<UCHAR>(signedValue));
                        }
        bool            operator<=(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress <= unsignedValue);
                        }
        bool            operator<=(const SHORT signedValue) const
                        {
                            return (m_ulAddress <= static_cast<USHORT>(signedValue));
                        }
        bool            operator<=(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress <= unsignedValue);
                        }
        bool            operator<=(const INT signedValue) const
                        {
                            return (m_ulAddress <= static_cast<UINT>(signedValue));
                        }
        bool            operator<=(const UINT unsignedValue) const
                        {
                            return (m_ulAddress <= unsignedValue);
                        }
        bool            operator<=(const LONG signedValue) const
                        {
                            return (m_ulAddress <= static_cast<ULONG>(signedValue));
                        }
        bool            operator<=(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress <= unsignedValue);
                        }
        bool            operator<=(const LONG64 signedValue) const
                        {
                            return (m_ulAddress <= static_cast<ULONG64>(signedValue));
                        }
        bool            operator<=(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress <= unsignedValue);
                        }
        bool            operator<=(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress <= targetAddress.addr());
                        }

        bool            operator>=(const CHAR signedValue) const
                        {
                            return (m_ulAddress >= static_cast<UCHAR>(signedValue));
                        }
        bool            operator>=(const UCHAR unsignedValue) const
                        {
                            return (m_ulAddress >= unsignedValue);
                        }
        bool            operator>=(const SHORT signedValue) const
                        {
                            return (m_ulAddress >= static_cast<USHORT>(signedValue));
                        }
        bool            operator>=(const USHORT unsignedValue) const
                        {
                            return (m_ulAddress >= unsignedValue);
                        }
        bool            operator>=(const INT signedValue) const
                        {
                            return (m_ulAddress >= static_cast<UINT>(signedValue));
                        }
        bool            operator>=(const UINT unsignedValue) const
                        {
                            return (m_ulAddress >= unsignedValue);
                        }
        bool            operator>=(const LONG signedValue) const
                        {
                            return (m_ulAddress >= static_cast<ULONG>(signedValue));
                        }
        bool            operator>=(const ULONG unsignedValue) const
                        {
                            return (m_ulAddress >= unsignedValue);
                        }
        bool            operator>=(const LONG64 signedValue) const
                        {
                            return (m_ulAddress >= static_cast<ULONG64>(signedValue));
                        }
        bool            operator>=(const ULONG64 unsignedValue) const
                        {
                            return (m_ulAddress >= unsignedValue);
                        }
        bool            operator>=(const TARGET_ADDR<T>& targetAddress) const
                        {
                            return (m_ulAddress >= targetAddress.addr());
                        }

}; // class TARGET_ADDR

//******************************************************************************
//
//  Declare the different target address types
//
//******************************************************************************
typedef TARGET_ADDR<CpuVirtual>             CPU_VIRTUAL;
typedef TARGET_ADDR<CpuPhysical>            CPU_PHYSICAL;

//******************************************************************************
//
//  Class POINTER (Target system pointer [32/64-bit CPU virtual address])
//
//******************************************************************************
class POINTER
{
protected:
        CPU_VIRTUAL     m_ptrPointer;

public:
                        POINTER()                       { m_ptrPointer = 0; }
                        POINTER(ULONG64 ulPointer)      { m_ptrPointer = ulPointer; }
                        POINTER(CPU_VIRTUAL ptrPointer) { m_ptrPointer = ptrPointer; }

        ULONG64         pointer() const                 { return ((m_ptrPointer.addr() < 0x100000000) ? EXTEND(m_ptrPointer.addr()) : m_ptrPointer.addr()); }
        ULONG64         ptr() const                     { return TARGET(m_ptrPointer.addr()); }

        PULONG64        data()                          { return m_ptrPointer.data(); }

// Compound assignment operator overloads
const   POINTER&        operator+=(const CHAR signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const SHORT signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const USHORT unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const INT signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const UINT unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const LONG signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const ULONG unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const LONG64 signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   POINTER&        operator+=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }

const   POINTER&        operator-=(const CHAR signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const SHORT signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const USHORT unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const INT signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const UINT unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const LONG signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const ULONG unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const LONG64 signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   POINTER&        operator-=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }

const   POINTER&        operator&=(const CHAR signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const SHORT signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const USHORT unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const INT signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const UINT unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const LONG signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const ULONG unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const LONG64 signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   POINTER&        operator&=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }

const   POINTER&        operator|=(const CHAR signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const SHORT signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const USHORT unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const INT signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const UINT unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const LONG signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const ULONG unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const LONG64 signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   POINTER&        operator|=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }

const   POINTER&        operator^=(const CHAR signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const SHORT signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const USHORT unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const INT signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const UINT unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const LONG signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const ULONG unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const LONG64 signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   POINTER&        operator^=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
// Unary operator overloads
const   POINTER&        operator++()
                        {
                            ++m_ptrPointer;

                            return (*this);
                        }
const   POINTER&        operator++(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ptrPointer++;

                            return (*this);
                        }

const   POINTER&        operator--()
                        {
                            --m_ptrPointer;

                            return (*this);
                        }
const   POINTER&        operator--(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ptrPointer--;

                            return (*this);
                        }
        // Binary operator overloads
        POINTER         operator+(const CHAR signedValue) const
                        {
                            return (POINTER(*this) += signedValue);
                        }
        POINTER         operator+(const UCHAR unsignedValue) const
                        {
                            return (POINTER(*this) += unsignedValue);
                        }
        POINTER         operator+(const SHORT signedValue) const
                        {
                            return (POINTER(*this) += signedValue);
                        }
        POINTER         operator+(const USHORT unsignedValue) const
                        {
                            return (POINTER(*this) += unsignedValue);
                        }
        POINTER         operator+(const INT signedValue) const
                        {
                            return (POINTER(*this) += signedValue);
                        }
        POINTER         operator+(const UINT unsignedValue) const
                        {
                            return (POINTER(*this) += unsignedValue);
                        }
        POINTER         operator+(const LONG signedValue) const
                        {
                            return (POINTER(*this) += signedValue);
                        }
        POINTER         operator+(const ULONG unsignedValue) const
                        {
                            return (POINTER(*this) += unsignedValue);
                        }
        POINTER         operator+(const LONG64 signedValue) const
                        {
                            return (POINTER(*this) += signedValue);
                        }
        POINTER         operator+(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(*this) += unsignedValue);
                        }

        POINTER         operator-(const CHAR signedValue) const
                        {
                            return (POINTER(*this) -= signedValue);
                        }
        POINTER         operator-(const UCHAR unsignedValue) const
                        {
                            return (POINTER(*this) -= unsignedValue);
                        }
        POINTER         operator-(const SHORT signedValue) const
                        {
                            return (POINTER(*this) -= signedValue);
                        }
        POINTER         operator-(const USHORT unsignedValue) const
                        {
                            return (POINTER(*this) -= unsignedValue);
                        }
        POINTER         operator-(const INT signedValue) const
                        {
                            return (POINTER(*this) -= signedValue);
                        }
        POINTER         operator-(const UINT unsignedValue) const
                        {
                            return (POINTER(*this) -= unsignedValue);
                        }
        POINTER         operator-(const LONG signedValue) const
                        {
                            return (POINTER(*this) -= signedValue);
                        }
        POINTER         operator-(const ULONG unsignedValue) const
                        {
                            return (POINTER(*this) -= unsignedValue);
                        }
        POINTER         operator-(const LONG64 signedValue) const
                        {
                            return (POINTER(*this) -= signedValue);
                        }
        POINTER         operator-(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(*this) -= unsignedValue);
                        }
        ULONG64         operator-(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer - ptrPointer.m_ptrPointer);
                        }

        ULONG64         operator*(const CHAR signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const SHORT signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const INT signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const LONG signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }

        ULONG64         operator/(const CHAR signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const SHORT signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const INT signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const LONG signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }

        ULONG64         operator%(const CHAR signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const SHORT signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const INT signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const LONG signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        // Bitwise operator overloads
        POINTER         operator~()
                        {
                            m_ptrPointer = ~m_ptrPointer;

                            return (*this);
                        }

        POINTER         operator&(const CHAR signedValue) const
                        {
                            return (POINTER(*this) &= signedValue);
                        }
        POINTER         operator&(const UCHAR unsignedValue) const
                        {
                            return (POINTER(*this) &= unsignedValue);
                        }
        POINTER         operator&(const SHORT signedValue) const
                        {
                            return (POINTER(*this) &= signedValue);
                        }
        POINTER         operator&(const USHORT unsignedValue) const
                        {
                            return (POINTER(*this) &= unsignedValue);
                        }
        POINTER         operator&(const INT signedValue) const
                        {
                            return (POINTER(*this) &= signedValue);
                        }
        POINTER         operator&(const UINT unsignedValue) const
                        {
                            return (POINTER(*this) &= unsignedValue);
                        }
        POINTER         operator&(const LONG signedValue) const
                        {
                            return (POINTER(*this) &= signedValue);
                        }
        POINTER         operator&(const ULONG unsignedValue) const
                        {
                            return (POINTER(*this) &= unsignedValue);
                        }
        POINTER         operator&(const LONG64 signedValue) const
                        {
                            return (POINTER(*this) &= signedValue);
                        }
        POINTER         operator&(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(*this) &= unsignedValue);
                        }

        POINTER         operator|(const CHAR signedValue) const
                        {
                            return (POINTER(*this) |= signedValue);
                        }
        POINTER         operator|(const UCHAR unsignedValue) const
                        {
                            return (POINTER(*this) |= unsignedValue);
                        }
        POINTER         operator|(const SHORT signedValue) const
                        {
                            return (POINTER(*this) |= signedValue);
                        }
        POINTER         operator|(const USHORT unsignedValue) const
                        {
                            return (POINTER(*this) |= unsignedValue);
                        }
        POINTER         operator|(const INT signedValue) const
                        {
                            return (POINTER(*this) |= signedValue);
                        }
        POINTER         operator|(const UINT unsignedValue) const
                        {
                            return (POINTER(*this) |= unsignedValue);
                        }
        POINTER         operator|(const LONG signedValue) const
                        {
                            return (POINTER(*this) |= signedValue);
                        }
        POINTER         operator|(const ULONG unsignedValue) const
                        {
                            return (POINTER(*this) |= unsignedValue);
                        }
        POINTER         operator|(const LONG64 signedValue) const
                        {
                            return (POINTER(*this) |= signedValue);
                        }
        POINTER         operator|(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(*this) |= unsignedValue);
                        }

        POINTER         operator^(const CHAR signedValue) const
                        {
                            return (POINTER(*this) ^= signedValue);
                        }
        POINTER         operator^(const UCHAR unsignedValue) const
                        {
                            return (POINTER(*this) ^= unsignedValue);
                        }
        POINTER         operator^(const SHORT signedValue) const
                        {
                            return (POINTER(*this) ^= signedValue);
                        }
        POINTER         operator^(const USHORT unsignedValue) const
                        {
                            return (POINTER(*this) ^= unsignedValue);
                        }
        POINTER         operator^(const INT signedValue) const
                        {
                            return (POINTER(*this) ^= signedValue);
                        }
        POINTER         operator^(const UINT unsignedValue) const
                        {
                            return (POINTER(*this) ^= unsignedValue);
                        }
        POINTER         operator^(const LONG signedValue) const
                        {
                            return (POINTER(*this) ^= signedValue);
                        }
        POINTER         operator^(const ULONG unsignedValue) const
                        {
                            return (POINTER(*this) ^= unsignedValue);
                        }
        POINTER         operator^(const LONG64 signedValue) const
                        {
                            return (POINTER(*this) ^= signedValue);
                        }
        POINTER         operator^(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(*this) ^= unsignedValue);
                        }

        ULONG64         operator>>(const CHAR signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const SHORT signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const INT signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }

        ULONG64         operator<<(const CHAR signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const SHORT signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const INT signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const LONG signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        // Logical operator overloads
        bool            operator==(const CHAR signedValue) const
                        {
                            return (m_ptrPointer == static_cast<UCHAR>(signedValue));
                        }
        bool            operator==(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const SHORT signedValue) const
                        {
                            return (m_ptrPointer == static_cast<USHORT>(signedValue));
                        }
        bool            operator==(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const INT signedValue) const
                        {
                            return (m_ptrPointer == static_cast<UINT>(signedValue));
                        }
        bool            operator==(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const LONG signedValue) const
                        {
                            return (m_ptrPointer == static_cast<ULONG>(signedValue));
                        }
        bool            operator==(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer == static_cast<ULONG64>(signedValue));
                        }
        bool            operator==(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer == ptrPointer.m_ptrPointer);
                        }

        bool            operator!=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer != static_cast<UCHAR>(signedValue));
                        }
        bool            operator!=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer != static_cast<USHORT>(signedValue));
                        }
        bool            operator!=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const INT signedValue) const
                        {
                            return (m_ptrPointer != static_cast<UINT>(signedValue));
                        }
        bool            operator!=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const LONG signedValue) const
                        {
                            return (m_ptrPointer != static_cast<ULONG>(signedValue));
                        }
        bool            operator!=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer != static_cast<ULONG64>(signedValue));
                        }
        bool            operator!=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer != ptrPointer.m_ptrPointer);
                        }

        bool            operator>(const CHAR signedValue) const
                        {
                            return (m_ptrPointer > static_cast<UCHAR>(signedValue));
                        }
        bool            operator>(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const SHORT signedValue) const
                        {
                            return (m_ptrPointer > static_cast<USHORT>(signedValue));
                        }
        bool            operator>(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const INT signedValue) const
                        {
                            return (m_ptrPointer > static_cast<UINT>(signedValue));
                        }
        bool            operator>(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const LONG signedValue) const
                        {
                            return (m_ptrPointer > static_cast<ULONG>(signedValue));
                        }
        bool            operator>(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer > static_cast<ULONG64>(signedValue));
                        }
        bool            operator>(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer > ptrPointer.ptr());
                        }

        bool            operator<(const CHAR signedValue) const
                        {
                            return (m_ptrPointer < static_cast<UCHAR>(signedValue));
                        }
        bool            operator<(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const SHORT signedValue) const
                        {
                            return (m_ptrPointer < static_cast<USHORT>(signedValue));
                        }
        bool            operator<(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const INT signedValue) const
                        {
                            return (m_ptrPointer < static_cast<UINT>(signedValue));
                        }
        bool            operator<(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const LONG signedValue) const
                        {
                            return (m_ptrPointer < static_cast<ULONG>(signedValue));
                        }
        bool            operator<(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer < static_cast<ULONG64>(signedValue));
                        }
        bool            operator<(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer < ptrPointer.ptr());
                        }

        bool            operator<=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<UCHAR>(signedValue));
                        }
        bool            operator<=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<USHORT>(signedValue));
                        }
        bool            operator<=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const INT signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<UINT>(signedValue));
                        }
        bool            operator<=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const LONG signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<ULONG>(signedValue));
                        }
        bool            operator<=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<ULONG64>(signedValue));
                        }
        bool            operator<=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer <= ptrPointer.ptr());
                        }

        bool            operator>=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<UCHAR>(signedValue));
                        }
        bool            operator>=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<USHORT>(signedValue));
                        }
        bool            operator>=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const INT signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<UINT>(signedValue));
                        }
        bool            operator>=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const LONG signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<ULONG>(signedValue));
                        }
        bool            operator>=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<ULONG64>(signedValue));
                        }
        bool            operator>=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer >= ptrPointer.ptr());
                        }

                        operator CPU_VIRTUAL()          { return m_ptrPointer; }

}; // class POINTER

//******************************************************************************
//
//  Template TARGET_PTR (Target system pointer template)
//
//******************************************************************************
template <PointerType T>
class TARGET_PTR
{
private:
static  const PointerType m_Type = T;

protected:
        POINTER         m_ptrPointer;

public:
                        TARGET_PTR<T>()                     { m_ptrPointer = 0; }
                        TARGET_PTR<T>(POINTER ptrPointer)   { m_ptrPointer = ptrPointer; }
                        TARGET_PTR<T>(ULONG64 ptrPointer)   { m_ptrPointer = ptrPointer; }

        ULONG64         pointer() const                     { return ((m_ptrPointer.addr() < 0x100000000) ? EXTEND(m_ptrPointer.addr()) : m_ptrPointer.addr()); }
        ULONG64         ptr() const                         { return TARGET(m_ptrPointer.ptr()); }

        PULONG64        data()                              { return m_ptrPointer.data(); }

        PointerType     type() const                        { return m_Type; }

// Compound assignment operator overloads
const   TARGET_PTR<T>&  operator+=(const CHAR& signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const UCHAR& unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const SHORT& signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const USHORT& unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const INT& signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const UINT& unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const LONG& signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const ULONG& unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const LONG64& signedValue)
                        {
                            m_ptrPointer += signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator+=(const ULONG64& unsignedValue)
                        {
                            m_ptrPointer += unsignedValue;

                            return (*this);
                        }

const   TARGET_PTR<T>&  operator-=(const CHAR signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const SHORT signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const USHORT unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const INT signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const UINT unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const LONG signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const ULONG unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const LONG64 signedValue)
                        {
                            m_ptrPointer -= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator-=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer -= unsignedValue;

                            return (*this);
                        }

const   TARGET_PTR<T>&  operator&=(const CHAR signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const SHORT signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const USHORT unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const INT signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const UINT unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const LONG signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const ULONG unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const LONG64 signedValue)
                        {
                            m_ptrPointer &= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator&=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer &= unsignedValue;

                            return (*this);
                        }

const   TARGET_PTR<T>&  operator|=(const CHAR signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const SHORT signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const USHORT unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const INT signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const UINT unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const LONG signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const ULONG unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const LONG64 signedValue)
                        {
                            m_ptrPointer |= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator|=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer |= unsignedValue;

                            return (*this);
                        }

const   TARGET_PTR<T>&  operator^=(const CHAR signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const UCHAR unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const SHORT signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const USHORT unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const INT signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const UINT unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const LONG signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const ULONG unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const LONG64 signedValue)
                        {
                            m_ptrPointer ^= signedValue;

                            return (*this);
                        }
const   TARGET_PTR<T>&  operator^=(const ULONG64 unsignedValue)
                        {
                            m_ptrPointer ^= unsignedValue;

                            return (*this);
                        }
// Unary operator overloads
const   POINTER&        operator++()
                        {
                            ++m_ptrPointer;

                            return (*this);
                        }
const   POINTER&        operator++(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ptrPointer++;

                            return (*this);
                        }
const   POINTER&        operator--()
                        {
                            --m_ptrPointer;

                            return (*this);
                        }
const   POINTER&        operator--(INT unused)
                        {
                            UNREFERENCED_PARAMETER(unused);

                            m_ptrPointer--;

                            return (*this);
                        }
        // Binary operator overloads
        POINTER         operator+(const CHAR signedValue) const
                        {
                            return (POINTER(m_ptrPointer) += signedValue);
                        }
        POINTER         operator+(const UCHAR unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) += unsignedValue);
                        }
        POINTER         operator+(const SHORT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) += signedValue);
                        }
        POINTER         operator+(const USHORT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) += unsignedValue);
                        }
        POINTER         operator+(const INT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) += signedValue);
                        }
        POINTER         operator+(const UINT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) += unsignedValue);
                        }
        POINTER         operator+(const LONG signedValue) const
                        {
                            return (POINTER(m_ptrPointer) += signedValue);
                        }
        POINTER         operator+(const ULONG unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) += unsignedValue);
                        }
        POINTER         operator+(const LONG64 signedValue) const
                        {
                            return (POINTER(m_ptrPointer) += signedValue);
                        }
        POINTER         operator+(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) += unsignedValue);
                        }

        POINTER         operator-(const CHAR signedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= signedValue);
                        }
        POINTER         operator-(const UCHAR unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= unsignedValue);
                        }
        POINTER         operator-(const SHORT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= signedValue);
                        }
        POINTER         operator-(const USHORT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= unsignedValue);
                        }
        POINTER         operator-(const INT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= signedValue);
                        }
        POINTER         operator-(const UINT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= unsignedValue);
                        }
        POINTER         operator-(const LONG signedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= signedValue);
                        }
        POINTER         operator-(const ULONG unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= unsignedValue);
                        }
        POINTER         operator-(const LONG64 signedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= signedValue);
                        }
        POINTER         operator-(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) -= unsignedValue);
                        }
        ULONG64         operator-(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer - ptrPointer.ptr());
                        }
        ULONG64         operator-(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer - ptrPointer.m_ptrPointer);
                        }

        ULONG64         operator*(const CHAR signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const SHORT signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const INT signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const LONG signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }
        ULONG64         operator*(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer * signedValue);
                        }
        ULONG64         operator*(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer * unsignedValue);
                        }

        ULONG64         operator/(const CHAR signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const SHORT signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const INT signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const LONG signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }
        ULONG64         operator/(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer / signedValue);
                        }
        ULONG64         operator/(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer / unsignedValue);
                        }

        ULONG64         operator%(const CHAR signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const SHORT signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const INT signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const LONG signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        ULONG64         operator%(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer % signedValue);
                        }
        ULONG64         operator%(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer % unsignedValue);
                        }
        // Bitwise operator overloads
        POINTER         operator~()
                        {
                            m_ptrPointer = ~m_ptrPointer;

                            return (*this);
                        }

        POINTER         operator&(const CHAR signedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= signedValue);
                        }
        POINTER         operator&(const UCHAR unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= unsignedValue);
                        }
        POINTER         operator&(const SHORT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= signedValue);
                        }
        POINTER         operator&(const USHORT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= unsignedValue);
                        }
        POINTER         operator&(const INT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= signedValue);
                        }
        POINTER         operator&(const UINT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= unsignedValue);
                        }
        POINTER         operator&(const LONG signedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= signedValue);
                        }
        POINTER         operator&(const ULONG unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= unsignedValue);
                        }
        POINTER         operator&(const LONG64 signedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= signedValue);
                        }
        POINTER         operator&(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) &= unsignedValue);
                        }

        POINTER         operator|(const CHAR signedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= signedValue);
                        }
        POINTER         operator|(const UCHAR unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= unsignedValue);
                        }
        POINTER         operator|(const SHORT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= signedValue);
                        }
        POINTER         operator|(const USHORT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= unsignedValue);
                        }
        POINTER         operator|(const INT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= signedValue);
                        }
        POINTER         operator|(const UINT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= unsignedValue);
                        }
        POINTER         operator|(const LONG signedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= signedValue);
                        }
        POINTER         operator|(const ULONG unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= unsignedValue);
                        }
        POINTER         operator|(const LONG64 signedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= signedValue);
                        }
        POINTER         operator|(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) |= unsignedValue);
                        }

        POINTER         operator^(const CHAR signedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= signedValue);
                        }
        POINTER         operator^(const UCHAR unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= unsignedValue);
                        }
        POINTER         operator^(const SHORT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= signedValue);
                        }
        POINTER         operator^(const USHORT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= unsignedValue);
                        }
        POINTER         operator^(const INT signedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= signedValue);
                        }
        POINTER         operator^(const UINT unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= unsignedValue);
                        }
        POINTER         operator^(const LONG signedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= signedValue);
                        }
        POINTER         operator^(const ULONG unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= unsignedValue);
                        }
        POINTER         operator^(const LONG64 signedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= signedValue);
                        }
        POINTER         operator^(const ULONG64 unsignedValue) const
                        {
                            return (POINTER(m_ptrPointer) ^= unsignedValue);
                        }

        ULONG64         operator>>(const CHAR signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const SHORT signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const INT signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }
        ULONG64         operator>>(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer >> signedValue);
                        }
        ULONG64         operator>>(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer >> unsignedValue);
                        }

        ULONG64         operator<<(const CHAR signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const SHORT signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const INT signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const LONG signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        ULONG64         operator<<(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer << signedValue);
                        }
        ULONG64         operator<<(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer << unsignedValue);
                        }
        // Logical operator overloads
        bool            operator==(const CHAR signedValue) const
                        {
                            return (m_ptrPointer == static_cast<UCHAR>(signedValue));
                        }
        bool            operator==(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const SHORT signedValue) const
                        {
                            return (m_ptrPointer == static_cast<USHORT>(signedValue));
                        }
        bool            operator==(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const INT signedValue) const
                        {
                            return (m_ptrPointer == static_cast<UINT>(signedValue));
                        }
        bool            operator==(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const LONG signedValue) const
                        {
                            return (m_ptrPointer == static_cast<ULONG>(signedValue));
                        }
        bool            operator==(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer == static_cast<ULONG64>(signedValue));
                        }
        bool            operator==(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer == unsignedValue);
                        }
        bool            operator==(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer == ptrPointer.ptr());
                        }
        bool            operator==(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer == ptrPointer.m_ptrPointer);
                        }

        bool            operator!=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer != static_cast<UCHAR>(signedValue));
                        }
        bool            operator!=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer != static_cast<USHORT>(signedValue));
                        }
        bool            operator!=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const INT signedValue) const
                        {
                            return (m_ptrPointer != static_cast<UINT>(signedValue));
                        }
        bool            operator!=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const LONG signedValue) const
                        {
                            return (m_ptrPointer != static_cast<ULONG>(signedValue));
                        }
        bool            operator!=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer != static_cast<ULONG64>(signedValue));
                        }
        bool            operator!=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer != unsignedValue);
                        }
        bool            operator!=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer != ptrPointer.ptr());
                        }
        bool            operator!=(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer != ptrPointer.m_ptrPointer);
                        }

        bool            operator>(const CHAR signedValue) const
                        {
                            return (m_ptrPointer > static_cast<UCHAR>(signedValue));
                        }
        bool            operator>(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const SHORT signedValue) const
                        {
                            return (m_ptrPointer > static_cast<USHORT>(signedValue));
                        }
        bool            operator>(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const INT signedValue) const
                        {
                            return (m_ptrPointer > static_cast<UINT>(signedValue));
                        }
        bool            operator>(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const LONG signedValue) const
                        {
                            return (m_ptrPointer > static_cast<ULONG>(signedValue));
                        }
        bool            operator>(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer > static_cast<ULONG64>(signedValue));
                        }
        bool            operator>(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer > unsignedValue);
                        }
        bool            operator>(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer > ptrPointer.ptr());
                        }
        bool            operator>(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer > ptrPointer.m_ptrPointer);
                        }

        bool            operator<(const CHAR signedValue) const
                        {
                            return (m_ptrPointer < static_cast<UCHAR>(signedValue));
                        }
        bool            operator<(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const SHORT signedValue) const
                        {
                            return (m_ptrPointer < static_cast<USHORT>(signedValue));
                        }
        bool            operator<(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const INT signedValue) const
                        {
                            return (m_ptrPointer < static_cast<UINT>(signedValue));
                        }
        bool            operator<(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const LONG signedValue) const
                        {
                            return (m_ptrPointer < static_cast<ULONG>(signedValue));
                        }
        bool            operator<(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer < static_cast<ULONG64>(signedValue));
                        }
        bool            operator<(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer < unsignedValue);
                        }
        bool            operator<(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer < ptrPointer.ptr());
                        }
        bool            operator<(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer < ptrPointer.m_ptrPointer);
                        }

        bool            operator<=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<UCHAR>(signedValue));
                        }
        bool            operator<=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<USHORT>(signedValue));
                        }
        bool            operator<=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const INT signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<UINT>(signedValue));
                        }
        bool            operator<=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const LONG signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<ULONG>(signedValue));
                        }
        bool            operator<=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer <= static_cast<ULONG64>(signedValue));
                        }
        bool            operator<=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer <= unsignedValue);
                        }
        bool            operator<=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer <= ptrPointer.ptr());
                        }
        bool            operator<=(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer <= ptrPointer.m_ptrPointer);
                        }

        bool            operator>=(const CHAR signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<UCHAR>(signedValue));
                        }
        bool            operator>=(const UCHAR unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const SHORT signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<USHORT>(signedValue));
                        }
        bool            operator>=(const USHORT unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const INT signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<UINT>(signedValue));
                        }
        bool            operator>=(const UINT unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const LONG signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<ULONG>(signedValue));
                        }
        bool            operator>=(const ULONG unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const LONG64 signedValue) const
                        {
                            return (m_ptrPointer >= static_cast<ULONG64>(signedValue));
                        }
        bool            operator>=(const ULONG64 unsignedValue) const
                        {
                            return (m_ptrPointer >= unsignedValue);
                        }
        bool            operator>=(const POINTER& ptrPointer) const
                        {
                            return (m_ptrPointer >= ptrPointer.ptr());
                        }
        bool            operator>=(const TARGET_PTR<T>& ptrPointer) const
                        {
                            return (m_ptrPointer >= ptrPointer.m_ptrPointer);
                        }

                        operator POINTER()              { return m_ptrPointer; }

}; // class TARGET_PTR

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _POINTER_H
