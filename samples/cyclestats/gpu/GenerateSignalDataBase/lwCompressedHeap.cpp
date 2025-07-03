 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <windows.h>

#include <lw32.h>
#include <lwos.h>

#include "lwCompressedHeap.h"
#include "lwGenericHashTable.h"

//

LwCompressedHeap::LwCompressedHeap()
{
    m_base = NULL;
    m_pHashTable = NULL;
    m_bytesUsed = 0;
    m_bytesAllocated = 0;
    m_allowResize = false;
}

//

LwCompressedHeap::~LwCompressedHeap()
{
    if (m_base)
    {
        destroy();
    }
    assert(!m_base);
}

// create the first block (which is guaranteed to be at offset 0)

LwCompressedHeap *LwCompressedHeap::create(LwU32 maxHeapSizeInBytes, bool allowResize)
{
    LwCompressedHeap *pHeap = new LwCompressedHeap;
    if (pHeap)
    {
        pHeap->m_allowResize = allowResize;

        // attach a hashtable
        if (!pHeap->m_pHashTable)
        {
            pHeap->m_pHashTable = new LwHashTable();
            if (!pHeap->m_pHashTable || !pHeap->m_pHashTable->create(1, Fragment::hash, Fragment::compare, Fragment::release))
            {
                // OOM
                delete pHeap;
                return NULL;
            }
        }

        assert(!pHeap->m_base);
        if (maxHeapSizeInBytes)
        {
            pHeap->m_base = new char[maxHeapSizeInBytes];
            if (pHeap->m_base)
            {
                // memset rest
                memset(pHeap->m_base, 0, maxHeapSizeInBytes);
                pHeap->m_bytesAllocated = maxHeapSizeInBytes;
            }
            else
            {
                // OOM
                delete pHeap;
                pHeap = NULL;
            }
        }
    }
    return pHeap;
}

//

void LwCompressedHeap::destroy()
{
    if (m_base)
    {
        delete[] m_base;
        m_base = NULL;
        m_bytesUsed = 0;
        m_bytesAllocated = 0;
    }
    if (m_pHashTable)
    {
        m_pHashTable->destroy();
        delete m_pHashTable;
        m_pHashTable = NULL;
    }
    delete this;
}

void *LwCompressedHeap::makeSpace(LwU32 bytes)
{
    // align size to multiple of 32bit
    bytes = (bytes+3) & ~3;

    // grow?
    if ((m_bytesUsed + bytes) > m_bytesAllocated)
    {
        if (!m_allowResize)
        {
            return NULL;
        }

        // first allocation ever?
        LwU32 newSize = (2 * (m_bytesUsed + bytes) + 0xfff) & ~0xfff;
        void *pNewBase = new char[newSize];
        if (pNewBase)
        {
            // copy content
            if (m_base)
            {
                memcpy(pNewBase, m_base, m_bytesUsed);
            }

            // memset rest
            memset((char*)pNewBase + m_bytesUsed, 0, newSize-m_bytesUsed);

            // release old block
            if (m_base)
            {
                delete[] m_base;
            }
            
            // switch
            m_base = pNewBase;
            m_bytesAllocated = newSize;
        }
        else
        {
            assert(0);
            return NULL;
        }
    }

    assert(m_base);
    assert((m_bytesUsed+bytes) <= m_bytesAllocated);
    assert((m_bytesUsed & 3) == 0);

    // reserve the first block as is
    void *p = (char*)m_base + m_bytesUsed;
    m_bytesUsed += bytes;
    assert(m_bytesUsed <= m_bytesAllocated);
    return p;
}

//

bool LwCompressedHeap::isAddrInHeap(const void *pData)
{
    size_t offset = (char*)pData - (char*)m_base;
    return (offset < m_bytesAllocated);
}

//

LwU32 LwCompressedHeap::getOffsetDIV4(void *addr)
{
    assert(m_base);
    LwU32 offset = (LwU32) ((char*)addr - (char*)m_base);
    assert(offset < m_bytesAllocated);
    assert((offset & 3) == 0);
    return offset >> 2;
}

//

void *LwCompressedHeap::getBase()
{
    assert(m_base);
    return m_base;
}

//

LwU32 LwCompressedHeap::getTotalSize()
{
    assert(m_base);
    return m_bytesUsed;
}

// stores a string (in lowercase!!)

void *LwCompressedHeap::addRedundantString(const char *s)
{
    LwU32 l = (LwU32) strlen(s);
    char *p = new char[l+1];
    if (p)
    {
        // colwert to lowercase
        memcpy(p, s, l+1);
        _strlwr(p);

        // store "token"
        void *r = addRedundantData(p, l+1);

        // allow reuse from addRedundantTwoPartString()
        generateAllPrePostFixHashes(r, l+1);

        // release tmp space and return the new location
        delete[] p;
        return r;
    }
    else
    {
        assert(0);
    }
    return NULL;
}

// stores a string (in lowercase!!)

bool LwCompressedHeap::addRedundantTwoPartString(void **ppNewNamePart0, LwU32 *pNameLenPart0, void **ppNewNamePart1, LwU32 *pNameLenPart1, const char *s)
{
    // deal with null string now so we don't have to deal with it later
    LwU32 l = (LwU32) strlen(s);
    if (!l)
    {
        *ppNewNamePart0 = NULL;
        *pNameLenPart0  = 0;
        *ppNewNamePart1 = NULL;
        *pNameLenPart1  = 0;
        return true;
    }

    // colwert/store in lowercase
    char *p = new char[l+1];
    if (p)
    {
        memcpy(p, s, l+1);
        _strlwr(p);

        void *pNewPart0 = NULL;
        void *pNewPart1 = NULL;
        LwU32 lenPart0  = 0;
        LwU32 lenPart1  = 0;

        // see which parts already exit / start with longest possible prefix first
        for(LwU32 i=l; --i>0;)
        {
            lenPart0 = i;
            lenPart1 = l - lenPart0;

            char *pNamePart0 = p;
            char *pNamePart1 = p + lenPart0;

            // avoid very short prefixes, i.e. one character is not helpful.
            if (lenPart0 < (l/2))
            {
                break;
            }

            // store "token"
            void *r = isDataRedundant(pNamePart0, lenPart0);
            if (r)
            {
                pNewPart0 = lenPart0 ? addRedundantData(pNamePart0, lenPart0) : NULL;
                pNewPart1 = lenPart1 ? addRedundantData(pNamePart1, lenPart1) : NULL;
                if (pNewPart1)
                {
                    generateAllPrePostFixHashes(pNewPart1, lenPart1);
                }
                break;
            }
        }

        // no existing match
        if (!pNewPart0 && !pNewPart1)
        {
            lenPart0 = l;
            lenPart1 = 0;
            pNewPart0 = addRedundantData(p, lenPart0);
            pNewPart1 = NULL;
            if (pNewPart0)
            {
                generateAllPrePostFixHashes(pNewPart0, lenPart0);
            }
        }

        // release tmp space and return the new location
        delete[] p;

        if (pNewPart0 && (!lenPart1 || pNewPart1))
        {
            *ppNewNamePart0 = pNewPart0;
            *pNameLenPart0  = lenPart0;
            *ppNewNamePart1 = pNewPart1;
            *pNameLenPart1  = lenPart1;
            return true;
        }
    }
    assert(0);
    return false;
}

// is blob redundant?

void *LwCompressedHeap::isDataRedundant(const void *pData, LwU32 dataLenInBytes)
{
    // did we already add this data once? (we allow pointer aliasing / pointers are not unique)
    Fragment newFrag((void*)pData, dataLenInBytes);
    Fragment *oldFrag = NULL;
    if (m_pHashTable->get((void**)&oldFrag, &newFrag))
    {
        return oldFrag->getData();
    }
    return NULL;
}

// add new blob

void *LwCompressedHeap::addRedundantData(const void *pData, LwU32 dataLenInBytes)
{
    // did we already add this data once? (we allow pointer aliasing / pointers are not unique)
    void *pOldData = isDataRedundant(pData, dataLenInBytes);
    if (pOldData)
    {
        return pOldData;
    }

    // add to compressed heap
    void *pNewData = addUniqueData((void*)pData, dataLenInBytes);

    // add to hashtable so we can find it later
    if (pNewData)
    {
        Fragment *pFrag = new Fragment(pNewData, dataLenInBytes);
        if (pFrag)
        {
            if (!m_pHashTable->add(pFrag, pFrag))
            {
                assert(0);
            }
        }
        else
        {
            assert(0);
        }
    }

    return pNewData;
}

// creates hash hits for every possible prefix/postfix from addRedundantData()

void LwCompressedHeap::generateAllPrePostFixHashes(void *pData, LwU32 dataLenInBytes)
{
    // pointer must be from our heap, i.e. addRedundantData() result
    assert(isAddrInHeap(pData));

    for(LwU32 i=1; i<dataLenInBytes; i++)
    {
        // possible prefix
        if (!isDataRedundant(pData, i))
        {
            Fragment *pFrag = new Fragment(pData, i);
            if (pFrag)
            {
                if (!m_pHashTable->add(pFrag, pFrag))
                {
                    assert(0);
                }
            }
            else
            {
                assert(0);
            }
        }

        // possible postfix (must be properly aligned)
        if (((i & 3) == 0) && !isDataRedundant((LwU8*)pData+i, dataLenInBytes-i))
        {
            Fragment *pFrag = new Fragment((LwU8*)pData+i, dataLenInBytes-i);
            if (pFrag)
            {
                if (!m_pHashTable->add(pFrag, pFrag))
                {
                    assert(0);
                }
            }
            else
            {
                assert(0);
            }
        }
    }
}

//

void *LwCompressedHeap::addUniqueData(void *pData, LwU32 dataLenInBytes)
{
    void *p = makeSpace(dataLenInBytes);
    if (p)
    {
        if (pData)
        {
            memcpy(p, pData, dataLenInBytes);
        }
        else
        {
            memset(p, 0, dataLenInBytes);
        }
    }
    return p;
}

//

LwCompressedHeap::Fragment::Fragment(void *pData, LwU32 dataSize)
{
    m_pData = pData;
    m_dataSize = dataSize;
}

//

LwCompressedHeap::Fragment::~Fragment()
{
    // not our memory!
}

//

void *LwCompressedHeap::Fragment::getData()
{
    return m_pData;
}

//

LwU32 LwCompressedHeap::Fragment::getSize()
{
    return m_dataSize;
}

// [callback]

LwU32 LwCompressedHeap::Fragment::hash(void *pKey)
{
    LwCompressedHeap::Fragment *pArgs = (LwCompressedHeap::Fragment*)pKey;
    assert(pArgs->getSize());

    LwU32 hash = 0x23424345;
    LwU8 *pData = (LwU8*) pArgs->getData();
    for(LwU32 i=0; i<pArgs->getSize(); i++)
    {
        hash = (hash+i+1) * 13;
        hash += pData[i];
        hash = (hash+i+1) * 7;
        hash ^= pData[i];
    }
    return (LwU32)hash;
}

// [callback]

bool LwCompressedHeap::Fragment::compare(void *pKey0, void *pKey1)
{
    LwCompressedHeap::Fragment *pArgs0 = (LwCompressedHeap::Fragment*)pKey0;
    LwCompressedHeap::Fragment *pArgs1 = (LwCompressedHeap::Fragment*)pKey1;
    if ((pArgs0->getSize() == pArgs1->getSize()) && (memcmp(pArgs0->getData(), pArgs1->getData(), pArgs0->getSize()) == 0))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// [callback]

void LwCompressedHeap::Fragment::release(void *pKey0, void *pData)
{
    LwCompressedHeap::Fragment *pArgs = (LwCompressedHeap::Fragment*)pData;
    delete pArgs;
}
