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
|*  Module: smartptr.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SMARTPTR_H
#define _SMARTPTR_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define STATIC_REFERENCE    0xffffffff          // Static object "special" refernce count value

//******************************************************************************
//
//  Class CRefObj (Reference counted object)
//
//******************************************************************************
class CRefObj
{
private:
mutable ULONG           m_ReferenceCount;

protected:
        void            setStatic() const               { m_ReferenceCount = STATIC_REFERENCE; }

                        CRefObj()                       { m_ReferenceCount = 0; }
virtual                ~CRefObj()                       { }

public:
        ULONG           acquire() const                 
                        {
                            // Only reference count the non-static objects
                            if (m_ReferenceCount != STATIC_REFERENCE)
                            {
                                ++m_ReferenceCount;
                            }
                            return m_ReferenceCount;
                        }
        ULONG           release() const
                        {
                            assert(m_ReferenceCount);

                            // Only reference count the non-static objects
                            if (m_ReferenceCount != STATIC_REFERENCE)
                            {
                                if (--m_ReferenceCount == 0)
                                {
                                    delete this;

                                    return 0;
                                }
                            }
                            return m_ReferenceCount;
                        }
        ULONG           references() const              { return m_ReferenceCount; }

}; // class CRefObj

//******************************************************************************
//
//  Template CPtrRef (Wrapper to provide CPtr references)
//
//******************************************************************************
template <typename R>
class CPtrRef
{
public:
mutable const R*        m_pRef;

        explicit        CPtrRef<R>(const R* pRef = NULL){ m_pRef = pRef; }
virtual                ~CPtrRef<R>()                    {}

}; // CPtrRef

//******************************************************************************
//
//  Template CPtr (Smart pointer template)
//
//******************************************************************************
template <typename T>
class CPtr
{
protected:
mutable const T*        m_pPtr;

public:
        explicit        CPtr<T>(const T* pPtr = NULL)   { m_pPtr = pPtr; }
                        CPtr<T>(const CPtr<T>& Ptr)     { m_pPtr = Ptr.release(); }
        template <typename R> CPtr<T>(const CPtr<R>& Ptr)
                            { m_pPtr = Ptr.release(); }
virtual                ~CPtr<T>()                       { delete m_pPtr; }

const   CPtr<T>&        operator=(const CPtr<T>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }
const   CPtr<T>&        operator=(const T* pPtr) const
                        {
                            reset(pPtr);

                            return (*this);
                        }
        template <typename R> const CPtr<T>& operator=(const CPtr<R>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }

                        CPtr<T>(const CPtrRef<T> Ref)   { m_pPtr = Ref.m_pRef; }

const   CPtr<T>&        operator=(const CPtrRef<T> Ref) const
                        {
                            if (Ref.m_pRef != m_pPtr)
                            {
                                delete m_pPtr;
                                m_pPtr = Ref.m_pRef;
                            }
                            return (*this);
                        }
        template <typename R> operator CPtrRef<R>()
                        {
                            T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return (CPtrRef<R>(pTmp));
                        }

        template <typename R> operator CPtr<R>()
                        {
                            T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return (CPtr<R>(pTmp));
                        }

        bool            operator==(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr == Ptr.m_pPtr); }
        bool            operator==(const void *pPtr) const
                            { return (this->m_pPtr == pPtr); }

        bool            operator!=(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr != Ptr.m_pPtr); }
        bool            operator!=(const void *pPtr) const
                            { return (this->m_pPtr != pPtr); }

        bool            operator<=(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr <= Ptr.m_pPtr); }
        bool            operator<=(const void *pPtr) const
                            { return (this->m_pPtr <= pPtr); }

        bool            operator>=(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr >= Ptr.m_pPtr); }
        bool            operator>=(const void *pPtr) const
                            { return (this->m_pPtr >= pPtr); }

        bool            operator<(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr < Ptr.m_pPtr); }
        bool            operator<(const void *pPtr) const
                            { return (this->m_pPtr < pPtr); }

        bool            operator>(const CPtr<T>& Ptr) const
                            { return (this->m_pPtr > Ptr.m_pPtr); }
        bool            operator>(const void *pPtr) const
                            { return (this->m_pPtr > pPtr); }

const   T*              release() const
                        {
                            const T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return pTmp;
                        }
        void            reset(const T* pPtr = NULL) const
                        {
                            if (pPtr != m_pPtr)
                            {
                                delete m_pPtr;
                                m_pPtr = pPtr;
                            }
                        }            
    
const   T*              ptr() const                     { return m_pPtr; }
        T*              ptr()                           { return const_cast<T*>(m_pPtr); }

const   T&              operator*() const               { return (*m_pPtr); }
const   T*              operator->() const              { return m_pPtr; }

        T&              operator*()                     { return *(const_cast<T*>(m_pPtr)); }
        T*              operator->()                    { return const_cast<T*>(m_pPtr); }

                        operator const T*() const       { return m_pPtr; }

}; // template CPtr

//******************************************************************************
//
//  Template CArrayRef (Wrapper to provide CArrayPtr references)
//
//******************************************************************************
template <typename R>
class CArrayRef
{
public:
mutable const R*        m_pRef;

        explicit        CArrayRef<R>(const R* pRef = NULL)
                        {
                            m_pRef = pRef;
                        }
virtual                ~CArrayRef<R>()                  {}

}; // CArrayRef

//******************************************************************************
//
//  Template CArrayPtr (Array pointer template)
//
//******************************************************************************
template <typename T>
class CArrayPtr
{
protected:
mutable const T*        m_pPtr;

public:
        explicit        CArrayPtr<T>(const T* pPtr = NULL)
                        {
                            m_pPtr = pPtr;
                        }
                        CArrayPtr<T>(const CArrayPtr<T>& Ptr)
                        {
                            m_pPtr = Ptr.release();
                        }
        template <typename R> CArrayPtr<T>(const CArrayPtr<R>& Ptr)
                        {
                            m_pPtr = Ptr.release();
                        }
virtual                ~CArrayPtr<T>()                  { delete[] m_pPtr; }

const   CArrayPtr<T>&   operator=(const CArrayPtr<T>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }
const   CArrayPtr<T>&   operator=(const T* pPtr) const
                        {
                            reset(pPtr);

                            return (*this);
                        }
        template <typename R> const CArrayPtr<T>& operator=(const CArrayPtr<R>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }

                        CArrayPtr<T>(const CArrayRef<T> Ref)
                        {
                            m_pPtr = Ref.m_pRef; }

const   CArrayPtr<T>&   operator=(const CArrayRef<T> Ref) const
                        {
                            if (Ref.m_pRef != m_pPtr)
                            {
                                delete [] m_pPtr;
                                m_pPtr = Ref.m_pRef;
                            }
                            return (*this);
                        }

        template <typename R> operator CArrayRef<R>()
                        {
                            T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return (CArrayRef<R>(pTmp));
                        }

        template <typename R> operator CArrayPtr<R>()
                        {
                            T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return (CArrayPtr<R>(pTmp));
                        }

        bool            operator==(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr == Ptr.m_pPtr); }
        bool            operator==(const void *pPtr) const
                            { return (this->m_pPtr == pPtr); }

        bool            operator!=(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr != Ptr.m_pPtr); }
        bool            operator!=(const void *pPtr) const
                            { return (this->m_pPtr != pPtr); }

        bool            operator<=(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr <= Ptr.m_pPtr); }
        bool            operator<=(const void *pPtr) const
                            { return (this->m_pPtr <= pPtr); }

        bool            operator>=(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr >= Ptr.m_pPtr); }
        bool            operator>=(const void *pPtr) const
                            { return (this->m_pPtr >= pPtr); }

        bool            operator<(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr < Ptr.m_pPtr); }
        bool            operator<(const void *pPtr) const
                            { return (this->m_pPtr < pPtr); }

        bool            operator>(const CArrayPtr<T>& Ptr) const
                            { return (this->m_pPtr > Ptr.m_pPtr); }
        bool            operator>(const void *pPtr) const
                            { return (this->m_pPtr > pPtr); }

const   T*              release() const
                        {
                            const T* pTmp = m_pPtr;
                            m_pPtr  = NULL;

                            return pTmp;
                        }
        void            reset(const T* pPtr = NULL) const
                        {
                            if (pPtr != m_pPtr)
                            {
                                delete [] m_pPtr;
                                m_pPtr = pPtr;
                            }
                        }            
    
const   T*              ptr() const                     { return m_pPtr; }
        T*              ptr()                           { return const_cast<T*>(m_pPtr); }

const   T&              operator*() const               { return (*m_pPtr); }
const   T*              operator->() const              { return m_pPtr; }

        T&              operator*()                     { return *(const_cast<T*>(m_pPtr)); }
        T*              operator->()                    { return const_cast<T*>(m_pPtr); }

const   T&              operator[](INT nIndex) const
                            { return m_pPtr[nIndex]; }
const   T&              operator[](UINT uIndex) const
                            { return m_pPtr[uIndex]; }
const   T&              operator[](LONG lIndex) const
                            { return m_pPtr[lIndex]; }
const   T&              operator[](ULONG ulIndex) const
                            { return m_pPtr[ulIndex]; }

        T&              operator[](INT nIndex)
                            { return const_cast<T*>(m_pPtr)[nIndex]; }
        T&              operator[](UINT uIndex)
                            { return const_cast<T*>(m_pPtr)[uIndex]; }
        T&              operator[](LONG lIndex)
                            { return const_cast<T*>(m_pPtr)[lIndex]; }
        T&              operator[](ULONG ulIndex)
                            { return const_cast<T*>(m_pPtr)[ulIndex]; }
        T&              operator[](ULONG64 ulIndex)
                            { return const_cast<T*>(m_pPtr)[ulIndex]; }

                        operator const T*() const       { return m_pPtr; }
                        operator T*()                   { return const_cast<T*>(m_pPtr); }

}; // template CArrayPtr

//******************************************************************************
//
//  Template CRefPtr (Reference counted pointer template)
//
//******************************************************************************
template <typename T>
class CRefPtr
{
protected:
mutable const T*        m_pPtr;

public:
                        CRefPtr<T>(const T* pPtr = NULL)
                        {
                            m_pPtr = pPtr;

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                        }
                        CRefPtr<T>(const CRefPtr<T>& Ptr)
                        {   
                            m_pPtr = Ptr.m_pPtr;

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                        }
virtual                ~CRefPtr<T>()
                        {
                            if (m_pPtr)
                            {
                                m_pPtr->release();
                            }
                            m_pPtr = NULL;
                        }

const   CRefPtr<T>&     operator=(const T& Ptr) const
                        {   
                            if (m_pPtr)
                            {
                                m_pPtr->release();
                            }
                            m_pPtr = &Ptr;

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                            return (*this);
                        }
const   CRefPtr<T>&     operator=(const T* pPtr) const
                        {
                            if (m_pPtr)
                            {
                                m_pPtr->release();
                            }
                            m_pPtr = const_cast<T*>(pPtr);

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                            return (*this);
                        }
const   CRefPtr<T>&     operator=(const CRefPtr<T>& Ptr) const
                        {
                            if (this != &Ptr)
                            {
                                if (m_pPtr)
                                {
                                    m_pPtr->release();
                                }
                                m_pPtr = Ptr.m_pPtr;

                                if (m_pPtr)
                                {
                                    m_pPtr->acquire();
                                }
                            }
                            return (*this);
                        }

        bool            operator==(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr == Ptr.m_pPtr); }
        bool            operator==(const void *pPtr) const
                            { return (this->m_pPtr == pPtr); }

        bool            operator!=(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr != Ptr.m_pPtr); }
        bool            operator!=(const void *pPtr) const
                            { return (this->m_pPtr != pPtr); }

        bool            operator<=(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr <= Ptr.m_pPtr); }
        bool            operator<=(const void *pPtr) const
                            { return (this->m_pPtr <= pPtr); }

        bool            operator>=(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr >= Ptr.m_pPtr); }
        bool            operator>=(const void *pPtr) const
                            { return (this->m_pPtr >= pPtr); }

        bool            operator<(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr < Ptr.m_pPtr); }
        bool            operator<(const void *pPtr) const
                            { return (this->m_pPtr < pPtr); }

        bool            operator>(const CRefPtr<T>& Ptr) const
                            { return (this->m_pPtr > Ptr.m_pPtr); }
        bool            operator>(const void *pPtr) const
                            { return (this->m_pPtr > pPtr); }

        ULONG           acquire() const                 { return m_pPtr->acquire(); }        
        ULONG           release() const                 { return m_pPtr->release(); }

const   T*              ptr() const                     { return m_pPtr; }
        T*              ptr()                           { return const_cast<T*>(m_pPtr); }

const   T&              operator*() const               { return (*m_pPtr); }
const   T*              operator->() const              { return m_pPtr; }

        T&              operator*()                     { return const_cast<T&>(*m_pPtr); }
        T*              operator->()                    { return const_cast<T*>(m_pPtr); }

                        operator const T*() const       { return m_pPtr; }
                        operator T*()                   { return const_cast<T*>(m_pPtr); }

}; // template CRefPtr

//******************************************************************************
//
//  Template CDrvPtr (Derived smart pointer template)
//
//******************************************************************************
template <typename T, typename B>
class CDrvPtr : public B
{
public:
                        CDrvPtr<T, B>(const T* pPtr = NULL)
                            { m_pPtr = pPtr; }
                        CDrvPtr<T, B>(const CPtr<T>& Ptr)
                            { m_pPtr = Ptr.release(); }
        explicit        CDrvPtr<T, B>(const B& Ptr)
                            { m_pPtr = Ptr.release(); }
virtual                ~CDrvPtr<T, B>()                 {};

const   CDrvPtr<T, B>&  operator=(const CDrvPtr<T, B>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }
const   CDrvPtr<T, B>&  operator=(const CPtr<T>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }

const   T*              ptr() const                     { return static_cast<T*>(m_pPtr); }
        T*              ptr()                           { return const_cast<T*>(static_cast<T*>(m_pPtr)); }

const   T&              operator*() const               { return *(static_cast<T*>(m_pPtr)); }
const   T*              operator->() const              { return static_cast<T*>(m_pPtr); }

        T&              operator*()                     { return *(const_cast<T*>(static_cast<const T*>(m_pPtr))); }
        T*              operator->()                    { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

                        operator const T*() const       { return static_cast<const T*>(m_pPtr); }
                        operator T*() const             { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

}; // template CDrvPtr

//******************************************************************************
//
//  Template CDrvArrayPtr (Derived array pointer template)
//
//******************************************************************************
template <typename T, typename B>
class CDrvArrayPtr : public B
{
public:
                        CDrvArrayPtr<T, B>(const T* pPtr = NULL)
                        {
                            m_pPtr = pPtr;
                        }
                        CDrvArrayPtr<T, B>(const CArrayPtr<T>& Ptr)
                        {
                            m_pPtr = Ptr.release();
                        }
        explicit        CDrvArrayPtr<T, B>(const B& Ptr)
                        {
                            m_pPtr = Ptr.release();
                        }
virtual                ~CDrvArrayPtr<T, B>()            {};

const   CDrvArrayPtr<T, B>& operator=(const CDrvArrayPtr<T, B>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }
const   CDrvArrayPtr<T, B>& operator=(const CArrayPtr<T>& Ptr) const
                        {
                            reset(Ptr.release());

                            return (*this);
                        }

const   T*              ptr() const                     { return static_cast<T*>(m_pPtr); }
        T*              ptr()                           { return const_cast<T*>(static_cast<T*>(m_pPtr)); }

const   T&              operator*() const               { return *(static_cast<const T*>(m_pPtr)); }
const   T*              operator->() const              { return static_cast<T*>(m_pPtr); }

        T&              operator*()                     { return *(const_cast<T*>(static_cast<const T*>(m_pPtr))); }
        T*              operator->()                    { return const_cast<T*>(static_cast<T*>(m_pPtr)); }

const   T&              operator[](INT nIndex) const
                            { return static_cast<const T*>(m_pPtr)[nIndex]; }
const   T&              operator[](UINT uIndex) const
                            { return static_cast<const T*>(m_pPtr)[uIndex]; }
const   T&              operator[](LONG lIndex) const
                            { return static_cast<const T*>(m_pPtr)[lIndex]; }
const   T&              operator[](ULONG ulIndex) const
                            { return static_cast<const T*>(m_pPtr)[ulIndex]; }

        T&              operator[](INT nIndex)
                            { return const_cast<T*>(static_cast<const T*>(m_pPtr))[nIndex]; }
        T&              operator[](UINT uIndex)
                            { return const_cast<T*>(static_cast<const T*>(m_pPtr))[uIndex]; }
        T&              operator[](LONG lIndex)
                            { return const_cast<T*>(static_cast<constT*>(m_pPtr))[lIndex]; }
        T&              operator[](ULONG ulIndex)
                            { return const_cast<T*>(static_cast<const T*>(m_pPtr))[ulIndex]; }
        T&              operator[](ULONG64 ulIndex)
                            { return const_cast<T*>(static_cast<const T*>(m_pPtr))[ulIndex]; }

                        operator const T*() const       { return static_cast<const T*>(m_pPtr); }
                        operator T*()                   { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

}; // template CDrvArrayPtr

//******************************************************************************
//
//  Template CDrvRefPtr (Derived reference counted pointer template)
//
//******************************************************************************
template <typename T, typename B>
class CDrvRefPtr : public B
{
public:
                        CDrvRefPtr<T, B>(const T* pPtr = NULL)
                        {
                            m_pPtr = const_cast<T*>(pPtr);

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                        }
                        CDrvRefPtr<T, B>(const CRefPtr<T>& Ptr)
                        {   
                            m_pPtr = Ptr.m_pPtr;

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                        }
        explicit        CDrvRefPtr<T, B>(const B& Ptr)
                        {   
                            m_pPtr = Ptr.ptr();

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                        }
virtual                ~CDrvRefPtr<T, B>()              {};

const   CDrvRefPtr<T, B>& operator=(const CRefPtr<T>& Ptr) const
                        {
                            if (this != &Ptr)
                            {
                                if (m_pPtr)
                                {
                                    m_pPtr->release();
                                }
                                m_pPtr = Ptr.m_pPtr;

                                if (m_pPtr)
                                {
                                    m_pPtr->acquire();
                                }
                            }
                            return (*this);
                        }
const   CDrvRefPtr<T, B>& operator=(const T* pPtr) const
                        {
                            if (m_pPtr)
                            {
                                m_pPtr->release();
                            }
                            m_pPtr = const_cast<T*>(pPtr);

                            if (m_pPtr)
                            {
                                m_pPtr->acquire();
                            }
                            return (*this);
                        }
const   CDrvRefPtr<T, B>& operator=(const CDrvRefPtr<T, B>& Ptr) const
                        {
                            if (this != &Ptr)
                            {
                                if (m_pPtr)
                                {
                                    m_pPtr->release();
                                }
                                m_pPtr = Ptr.m_pPtr;

                                if (m_pPtr)
                                {
                                    m_pPtr->acquire();
                                }
                            }
                            return (*this);
                        }

const   T*              ptr() const                     { return static_cast<const T*>(m_pPtr); }
        T*              ptr()                           { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

const   T&              operator*() const               { return *(static_cast<const T*>(m_pPtr)); }
const   T*              operator->() const              { return static_cast<const T*>(m_pPtr); }

        T&              operator*()                     { return *(const_cast<T*>(static_cast<const T*>(m_pPtr))); }
        T*              operator->()                    { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

                        operator const T*() const       { return static_cast<const T*>(m_pPtr); }
                        operator T*()                   { return const_cast<T*>(static_cast<const T*>(m_pPtr)); }

}; // template CDrvRefPtr

//******************************************************************************
//
//  Template CObjList (Object list, used for unique referenced counted objects)
//
//******************************************************************************
template <typename T>
class CObjList
{
private:
        T*              m_pFirstObject;
        T*              m_pLastObject;
        ULONG           m_ulObjectCount;

public:
                        CObjList()
                        {
                            m_pFirstObject = NULL;
                            m_pLastObject  = NULL;
                            m_ulObjectCount = 0;
                        }
virtual                ~CObjList()                      { };

        T*              firstObject()                   { return m_pFirstObject; }
        void            firstObject(T* pFirstObject)    { m_pFirstObject = pFirstObject; }

        T*              lastObject()                    { return m_pLastObject; }
        void            lastObject(T* pLastObject)      { m_pLastObject = pLastObject; }

        ULONG           objectCount()                   { return m_ulObjectCount; }
        ULONG           incrementCount()                { return (++m_ulObjectCount); }
        ULONG           decrementCount()                { return (--m_ulObjectCount); }

}; // template CObjList

//******************************************************************************
//
//  Template LWnqObj (Unique reference counted object)
//
//******************************************************************************
template <typename T>
class LWnqObj : public CRefObj
{
private:
        T*              m_pPrevObject;
        T*              m_pNextObject;
        CObjList<T>*    m_pList;
        ULONG64         m_key;

        void            addObject(CObjList<T> *pList, T *pUnqObj)
                        {
                            assert(pList);
                            assert(pUnqObj);

                            // Check for adding the first object
                            if (pList->firstObject() == NULL)
                            {
                                // Set first and last object to this new object
                                pList->firstObject(pUnqObj);
                                pList->lastObject(pUnqObj);

                                // Set object pointers to no other objects
                                prevObject(NULL);
                                nextObject(NULL);
                            }
                            else    // Not adding the first object
                            {
                                // Add this object to the end of the list
                                prevObject(pList->lastObject());
                                nextObject(NULL);

                                pList->lastObject()->nextObject(pUnqObj);
                                pList->lastObject(pUnqObj);
                            }
                            // Increment the list object count
                            pList->incrementCount();
                        }
        void            delObject(CObjList<T> *pList, T *pUnqObj)
                        {
                            assert(pList);
                            assert(pUnqObj);

                            // Check for deleting the first object
                            if (pList->firstObject() == pUnqObj)
                            {
                                // Check for deleting the only object (First & Last)
                                if (pList->lastObject() != pUnqObj)
                                {
                                    // Delete the first object from the list
                                    pList->firstObject()->nextObject()->prevObject(NULL);
                                    pList->firstObject(pList->firstObject()->nextObject());
                                }
                                else    // Deleting the only object in the list
                                {
                                    // Clear the list pointers
                                    pList->firstObject(NULL);
                                    pList->lastObject(NULL);
                                }
                            }
                            else    // Not deleting the first object
                            {
                                // Check for deleting the last object
                                if (pList->lastObject() == pUnqObj)
                                {
                                    // Delete the last object from the list
                                    pList->lastObject()->prevObject()->nextObject(NULL);
                                    pList->lastObject(pList->lastObject()->prevObject());
                                }
                                else    // Not deleting the last object
                                {
                                    // Delete the object from the list
                                    pUnqObj->prevObject()->nextObject(pUnqObj->nextObject());
                                    pUnqObj->nextObject()->prevObject(pUnqObj->prevObject());
                                }
                            }
                            // Clear the object previous/next pointers
                            prevObject(NULL);
                            nextObject(NULL);

                            // Decrement the list object count
                            if (pList->decrementCount() == 0)
                            {
                                // List better be empty now
                                assert(pList->firstObject() == NULL);
                                assert(pList->lastObject()  == NULL);
                            }
                        }
protected:
                        LWnqObj(CObjList<T>* pList, ULONG64 key)
                        {
                            // Save the object list pointer
                            m_pList = pList;

                            // Save the unique object key
                            m_key = key;

                            // Add this object to the object list
                            addObject(pList, static_cast<T*>(this));
                        }
                        LWnqObj(CObjList<T>* pList, POINTER ptrPointer)
                        {
                            // Save the object list pointer
                            m_pList = pList;

                            // Save the unique object key
                            m_key = ptrPointer.ptr();

                            // Add this object to the object list
                            addObject(pList, static_cast<T*>(this));
                        }
virtual                ~LWnqObj()
                        {
                            // Delete this object from the object list
                            delObject(m_pList, static_cast<T*>(this));
                        }

        ULONG64         key() const                     { return m_key; }

static  T*              findObject(CObjList<T>* pList, ULONG64 key)
                        {
                            T              *pUnqObj = pList->firstObject();

                            // Search all the existing objects for a matching key
                            while (pUnqObj != NULL)
                            {
                                // Check for a key match
                                if (pUnqObj->key() == key)
                                {
                                    break;
                                }
                                // Move to the next object
                                pUnqObj = pUnqObj->nextObject();
                            }
                            return pUnqObj;
                        }
static  T*              findObject(CObjList<T>* pList, POINTER ptrPointer)
                        {
                            T              *pUnqObj = pList->firstObject();

                            // Search all the existing objects for a matching key
                            while (pUnqObj != NULL)
                            {
                                // Check for a key match
                                if (pUnqObj->key() == ptrPointer.ptr())
                                {
                                    break;
                                }
                                // Move to the next object
                                pUnqObj = pUnqObj->nextObject();
                            }
                            return pUnqObj;
                        }
        T*              firstObject() const             { return (m_pList->firstObject()); }
        T*              lastObject() const              { return (m_pList->lastObject()); }
        ULONG           objectCount() const             { return (m_pList->objectCount()); }

        T*              prevObject() const              { return m_pPrevObject; }
        void            prevObject(T* pPrevObject)      { m_pPrevObject = pPrevObject; }
        T*              nextObject() const              { return m_pNextObject; }
        void            nextObject(T* pNextObject)      { m_pNextObject = pNextObject; }

public:
        T*              createObject(CObjList<T>* pList, ULONG64 key)
                        {
                            T              *pUnqObj = NULL;

                            // Try to find an object with this key
                            pUnqObj = findObject(pList, key);
                            if (pUnqObj != NULL)
                            {
                                // Simply acquire a reference to the already existing object
                                pUnqObj->acquire();
                            }
                            else    // Could not find this object
                            {
                                // Try to create the new object
                                pUnqObj = new LWnqObj(pList, key);
                            }
                            return pUnqObj;
                        }

}; // template LWnqObj

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SMARTPTR_H
