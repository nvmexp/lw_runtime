 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#if !defined(_H_LWCOMPRESSED_HEAP_H)
#define _H_LWCOMPRESSED_HEAP_H

class LwHashTable;

class LwCompressedHeap
{
    private:
        void                       *m_base;
        LwU32                       m_bytesUsed; 
        LwU32                       m_bytesAllocated; 
        bool                        m_allowResize : 1;
        LwHashTable                *m_pHashTable;

        class Fragment
        {
            private:
                void               *m_pData;
                LwU32               m_dataSize;

            public:
                                    Fragment(void *pData, LwU32 dataSize);
                                   ~Fragment();

                       void        *getData();
                       LwU32        getSize();

                static LwU32        hash(void *pKey);
                static bool         compare(void *pKey0, void *pKey1);
                static void         release(void *pKey0, void *pData);
        };

    private:
                                    LwCompressedHeap();
                                   ~LwCompressedHeap();
        void                       *makeSpace(LwU32 bytes);
        bool                        isAddrInHeap(const void *pData);
        void                       *isDataRedundant(const void *pData, LwU32 dataLenInBytes);
        void                        generateAllPrePostFixHashes(void *pData, LwU32 dataLenInBytes);

    public:
        static LwCompressedHeap    *create(LwU32 maxHeapSizeInBytes, bool allowResize);
               void                 destroy();

        void                       *addRedundantString(const char *s);
        bool                        addRedundantTwoPartString(void **ppNewNamePart0, LwU32 *pNameLenPart0, void **ppNewNamePart1, LwU32 *pNameLenPart1, const char *s);
        void                       *addRedundantData(const void *pData, LwU32 dataLenInBytes);
        void                       *addUniqueData(void *pData, LwU32 dataLenInBytes);
        void                       *getBase();
        LwU32                       getOffsetDIV4(void *addr);
        LwU32                       getTotalSize();
};

#endif // defined(_H_LWCOMPRESSED_HEAP_H)
