 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#if !defined(_H_LW_GENERIC_HASHTABLE_H_)
#define _H_LW_GENERIC_HASHTABLE_H_                         1

// implements a hashtable

class LwHashTable
{
    private:
        // each bucket is a single linked list of nodes
        class CollisionNode
        {
            public:
                // next collision in the current bucket
                CollisionNode       *pNext;

                // prev/next for enums
                CollisionNode       *pNextEnum;
                CollisionNode       *pPrevEnum;

                LwU32                hash;
                void                *pKey;
                void                *pData;
        };

        // buckets
        CollisionNode  **m_pHashTable;

        // all nodes are part of linked list as well
        CollisionNode   *m_pFirstEnum;

        // #buckets allocated
        LwU32            m_tableSize;

        // elements in the hash table itself
        LwU32            m_entryCount;

        // grow threshold
        LwU32            m_loadLimit;

        // callback/obtain hash value for a given key
        LwU32          (*m_hash)(void *pKey);

        // callback/compare to keys
        bool           (*m_compare)(void *pKey0, void *pKey1);

        // callback/release resource assiciated with the data
        void           (*m_release)(void *pKey, void *pData);

    public:
        // opaque iterator object (really is just a CollisionNode)
        class Iterator
        {
            public:
                void     *getData();
                Iterator *next();
        };

    private:
        LwU32       hashToIndex(LwU32 hash);
        bool        grow();

    public:
                    LwHashTable();
                   ~LwHashTable();
                   
        // creation/etc
        bool        create(LwU32 initialTableSize, LwU32 (*hash)(void *pKey), bool (*compare)(void *pKey0, void *pKey1), void (*release)(void *pKey, void *pData));
        bool        add(void *pKey, void *pData);
        bool        get(void **ppData, void *key);
        void        remove(void *key);
        void        destroy();

        // iterator support
        Iterator   *first();

        // accessors
        LwU32       getEntryCount() const;
};

#endif // defined(_H_LW_GENERIC_HASHTABLE_H_)
