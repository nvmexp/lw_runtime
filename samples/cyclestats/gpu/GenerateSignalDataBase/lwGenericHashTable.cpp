 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <windows.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lw32.h>
#include <lwos.h>

#include "lwGenericHashTable.h"

LwHashTable::LwHashTable()
{
    // (try to) leave this empty
    m_pHashTable = NULL;
    m_pFirstEnum = NULL;
}

LwHashTable::~LwHashTable()
{
    // (try to) leave this empty
    assert(!m_pHashTable);
    assert(!m_pFirstEnum);
}

//---------------------------------------------------------------------------

// create a new hashtable given an initial size and hash/compare functions

bool LwHashTable::create
(
    LwU32   initialTableSize,
    LwU32 (*hash)(void *pKey),
    bool  (*compare)(void *pKey0, void *pKey1),
    void  (*release)(void *pKey, void *pData))
{
    // note that the release callback is optional (release==NULL => no callback)
    assert(hash && compare);

    // common hash table sizes that uses primes (close to P2 values, ~doubles every time)
    static const LwU32 primes[] =
    {
        7, 13, 29, 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317,
        196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917, 25165843,
        50331653, 100663319, 201326611, 402653189, 805306457, 1610612741
    };

    // can create/grow past the last prime we got...
    LwU32 maxTableSize = primes[(sizeof(primes)/sizeof(primes[0]))-1];
    if (initialTableSize >= maxTableSize)
    {
        return false;
    }

    // pick closest one
    LwU32 tableSize = 0;
    for (LwU32 i=0; i<sizeof(primes)/sizeof(primes[0]); i++)
    {
        if (primes[i] > initialTableSize)
        {
            tableSize = primes[i];
            break;
        }
    }

    // save the arguments
    m_entryCount = 0;
    m_tableSize  = tableSize;
    m_loadLimit  = (tableSize/2)+1;
    m_pFirstEnum = NULL;

    m_hash       = hash;
    m_compare    = compare;
    m_release    = release;

    // allocate the table itself
    m_pHashTable = new CollisionNode*[m_tableSize];
    if (m_pHashTable)
    {
        memset(m_pHashTable, 0, sizeof(m_pHashTable[0]) * m_tableSize);
        return true;
    }
    return false;
}

// releases all resources

void LwHashTable::destroy()
{
    if (m_pHashTable)
    {
        // for each bucket
        for(LwU32 i=0; i<m_tableSize; i++)
        {
            for(CollisionNode *pNode = m_pHashTable[i]; pNode;)
            {
                CollisionNode *pNext = pNode->pNext;
                if (m_release)
                {
                    m_release(pNode->pKey, pNode->pData);
                }
                delete pNode;
                pNode = pNext;
            }
        }
        delete[] m_pHashTable;
        m_pHashTable = NULL;
    }
    m_pFirstEnum = NULL;
}

//---------------------------------------------------------------------------

// helper func to translate hash value into a starting index

LwU32 LwHashTable::hashToIndex(LwU32 hash)
{
    hash = hash+1;
    hash = hash * hash;
    return hash % m_tableSize;
}

//---------------------------------------------------------------------------

// expands the hashtable (to the next prime)

bool LwHashTable::grow()
{
    LwHashTable newHT;
    if (newHT.create(m_tableSize, m_hash, m_compare, m_release))
    {
        // for each bucket
        for(LwU32 i=0; i<m_tableSize; i++)
        {
            // for each node
            for(CollisionNode *pNode = m_pHashTable[i]; pNode;)
            {
                CollisionNode *pNext = pNode->pNext;

                // relink to the new table
                LwU32 newIndex = newHT.hashToIndex(pNode->hash);
                pNode->pNext = newHT.m_pHashTable[newIndex];
                newHT.m_pHashTable[newIndex] = pNode;

                pNode = pNext;
            }
            m_pHashTable[i] = NULL;
        }

        // swap tables/etc
        delete[] m_pHashTable;
        m_pHashTable       = newHT.m_pHashTable;
        m_tableSize        = newHT.m_tableSize;
        m_loadLimit        = newHT.m_loadLimit;
        newHT.m_pHashTable = NULL;
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------

// adds a _new_ (pKey, pData) mapping

bool LwHashTable::add(void *pKey, void *pData)
{
    assert(m_pHashTable);
    assert(!get(NULL, pKey));

    // grow?
    if (m_entryCount > m_loadLimit)
    {
        // we don't care if grow() fails - we just get longer collision lists
        grow();
    }

    // add new entry
    CollisionNode *pNew = new CollisionNode();
    if (pNew)
    {
        LwU32 hash = m_hash(pKey);
        LwU32 index = hashToIndex(hash);

        // make the new node first in the bucket
        pNew->pKey          = pKey;
        pNew->hash          = hash;
        pNew->pData         = pData;
        pNew->pNext         = m_pHashTable[index];
        m_pHashTable[index] = pNew;
        m_entryCount++;

        // ..and the enumlist
        pNew->pPrevEnum     = NULL;
        pNew->pNextEnum     = m_pFirstEnum;
        if (m_pFirstEnum)
        {
            m_pFirstEnum->pPrevEnum = pNew;
        }
        m_pFirstEnum        = pNew;

        // self consistency...
        assert(get(NULL, pKey));
        return true;
    }
    else
    {
        return false;
    }
}

//---------------------------------------------------------------------------

// retrieves the pData for element pKey

bool LwHashTable::get(void **ppData, void *pKey)
{
    assert(m_pHashTable);

    // search them all
    LwU32 hash = m_hash(pKey);
    LwU32 startIndex = hashToIndex(hash);
    CollisionNode *pPreNode = NULL;
    for(CollisionNode *pNode = m_pHashTable[startIndex]; pNode; pNode = pNode->pNext)
    {
        // match?
        if ((pNode->hash == hash) && (m_compare(pNode->pKey, pKey)))
        {
            if (ppData)
            {
                *ppData = pNode->pData;
            }

            // make it the first in the collision list
            if (pPreNode)
            {
                pPreNode->pNext          = pNode->pNext;
                pNode->pNext             = m_pHashTable[startIndex];
                m_pHashTable[startIndex] = pNode;
            }
            return true;
        }
        pPreNode = pNode;
    }
    if (ppData)
    {
        *ppData = NULL;
    }
    return false;
}

//---------------------------------------------------------------------------

// remove the element pKey

void LwHashTable::remove(void *pKey)
{
    // a NULL pKey is perfectly valid, m_hash must be able to deal w/ this case
    assert(m_pHashTable);

    // search and destroy...
    LwU32 hash = m_hash(pKey);
    LwU32 startIndex = hashToIndex(hash);
    CollisionNode *pPreNode = NULL;
    for(CollisionNode *pNode = m_pHashTable[startIndex]; pNode; pNode = pNode->pNext)
    {
        // match?
        if ((pNode->hash == hash) && (m_compare(pNode->pKey, pKey)))
        {
            // skip over this node
            if (pPreNode)
            {
                pPreNode->pNext = pNode->pNext;
            }
            else
            {
                // new root
                m_pHashTable[startIndex] = pNode->pNext;
            }

            // fix up enum list, too
            if (pNode->pNextEnum)
            {
                assert(pNode->pNextEnum->pPrevEnum == pNode);
                pNode->pNextEnum->pPrevEnum = pNode->pPrevEnum;
            }
            if (pNode->pPrevEnum)
            {
                assert(pNode->pPrevEnum->pNextEnum == pNode);
                pNode->pPrevEnum->pNextEnum = pNode->pNextEnum;
            }
            if (m_pFirstEnum == pNode)
            {
                m_pFirstEnum = pNode->pNextEnum;
            }

            // release the node itself
            if (m_release)
            {
                m_release(pNode->pKey, pNode->pData);
            }
            delete pNode;

            // no shrink atm...
            assert(m_entryCount);
            m_entryCount--;
            return;
        }
        pPreNode = pNode;
    }

    // not found!
    assert(0);
}

//---------------------------------------------------------------------------

// accessors

LwU32 LwHashTable::getEntryCount() const
{
    return m_entryCount;
}

//---------------------------------------------------------------------------

// iterator support / get a link to the first node

LwHashTable::Iterator *LwHashTable::first()
{
    return (LwHashTable::Iterator*)m_pFirstEnum;
}

// iterator support / move to the next node

LwHashTable::Iterator *LwHashTable::Iterator::next()
{
    CollisionNode *it = (CollisionNode*) this;
    return (LwHashTable::Iterator*)it->pNextEnum;
}

// iterator support / get the pData for the current iterator poistion

void *LwHashTable::Iterator::getData()
{
    CollisionNode *it = (CollisionNode*) this;
    return it->pData;
}
