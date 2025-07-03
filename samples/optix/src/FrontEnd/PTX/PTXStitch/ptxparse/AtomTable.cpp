// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES 

#include <AtomTable.h>
#include <ptxIR.h>

#ifdef _WINBASE_
#undef AddAtom
#endif

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// class StringTable: ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/*
 * StringTable::StringTable() - StringTable constructor.
 *
 */

StringTable::StringTable(IMemPool *fMemPool, int fsize)
{
    memPool = fMemPool;
    strings = (char*)newFromPool(fMemPool, INIT_STRING_TABLE_SIZE);
    // Zero-th offset means "empty" so don't use it.
    nextFree = 1;
    size = INIT_STRING_TABLE_SIZE;
} // StringTable::StringTable

/*
 * StringTable::Clone() - copy a Stringtable.
 *
 */

StringTable *StringTable::Clone(IMemPool *fMemPool) const
{
    StringTable *newTable = (StringTable*)newObjectFromPool<StringTable, IMemPool*>(fMemPool, fMemPool);
    newTable->memPool = fMemPool;
    newTable->nextFree = nextFree;
    newTable->size = size;
    newTable->strings = (char*)newFromPool(fMemPool, size);
    memcpy(newTable->strings, strings, size);
    return newTable;
} // StringTable::Clone


/*
 * HashString() - Hash a string with the base hash function.
 *
 */

static int HashString(const char *s)
{
    int hval = 0;

    while (*s) {
        hval = (hval*13507 + *s*197) ^ (hval >> 2);
        s++;
    }
    return hval & 0x7fffffff;
} // HashString

/*
 * HashString2() - Hash a string with the incrimenting hash function.
 *
 */

static int HashString2(const char *s)
{
    int hval = 0;

    while (*s) {
        hval = (hval*729 + *s*37) ^ (hval >> 1);
        s++;
    }
    return hval;
} // HashString2

/*
 * AddString() - Add a string to a string table.  Return it's offset.
 *
 */

static int AddString(StringTable *stable, const char *s)
{
    int len, loc;
    char *str;

    len = (int) strlen(s);
    if (stable->nextFree + len + 1 >= stable->size) {
        RT_ASSERT(stable->size < 64000000);
        str = (char*)newFromPool(stable->memPool, stable->size*2);
        memcpy(str, stable->strings, stable->size);
        //////////////////////////////////////////////////free(stable->strings);
        stable->strings = str;
        stable->size *= 2;
    }
    loc = stable->nextFree;
    strcpy(&stable->strings[loc], s);
    stable->nextFree += len + 1;
    return loc;
} // AddString

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Class HashTable: ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/*
 * HashTable::HashTable() - Hashtable constructor.
 *
 */

HashTable::HashTable(IMemPool *fMemPool, int fsize)
{
    int ii;

    memPool = fMemPool;
    entry = (HashEntry*)newFromPool(fMemPool, sizeof(HashEntry)*fsize); // new HashEntry[fsize]
    size = fsize;
    for (ii = 0; ii < fsize; ii++) {
        entry[ii].index = 0;
        entry[ii].value = 0;
    }
    entries = 0;
    for (ii = 0; ii <= HASH_TABLE_MAX_COLLISIONS; ii++)
        counts[ii] = 0;
} // HashTable::HashTable

/*
 * HashTable::Clone() - copy a Hashtable.
 *
 */

HashTable *HashTable::Clone(IMemPool *fMemPool) const
{
    HashTable *newTable = (HashTable*)newObjectFromPool<HashTable, IMemPool*, int>(fMemPool, fMemPool, size);
    newTable->entries = entries;
    memcpy(newTable->entry, entry, size * sizeof(HashEntry));
    memcpy(newTable->counts, counts,
           (HASH_TABLE_MAX_COLLISIONS+1) * sizeof(int));
    return newTable;
} // HashTable::Clone

/*
 * Empty() - See if a hash table entry is empty.
 *
 */

static int Empty(HashTable *htable, int hashloc)
{
    RT_ASSERT(hashloc >= 0 && hashloc < htable->size);
    if (htable->entry[hashloc].index == 0) {
        return 1;
    } else {
        return 0;
    }
} // Empty

/*
 * Match() - See if a hash table entry is matches a string.
 *
 */

static int Match(HashTable *htable, StringTable *stable, const char *s, int hashloc)
{
    int strloc;

    strloc = htable->entry[hashloc].index;
    if (!strcmp(s, &stable->strings[strloc])) {
        return 1;
    } else {
        return 0;
    }
} // Match

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Class AtomTable: ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

//static AtomTable latable = { { NULL, 0, 0 }, { NULL, 0, 0, { 0, 0, 0, 0} },
//                             NULL, NULL, 0, 0, NULL, 0 };
//AtomTable *atable = &latable;

static int AtomTable_addAtom(IAtomTable *atable, const char *fStr)
{
    return static_cast<AtomTable *>(atable)->LookUpAddString(fStr);
}

static const char *AtomTable_getString(IAtomTable *atable, int atom)
{
    return static_cast<AtomTable *>(atable)->GetAtomString(atom);
}

static int AtomTable_lookupAtom(IAtomTable *atable, const char *fStr)
{
    return static_cast<AtomTable *>(atable)->LookUpString(fStr);
}

static IAtomTable_ops AtomTable_ops = {
    AtomTable_addAtom,
    AtomTable_getString,
    AtomTable_lookupAtom,
};

/*
 * AtomTable::AtomTable() - Atom table constructor.
 *
 */

AtomTable::AtomTable(IMemPool *fMemPool, int htsize) //: stable(fMemPool)
{
    ops = &AtomTable_ops;
    htsize = htsize <= 0 ? INIT_HASH_TABLE_SIZE : htsize;
    htable = (HashTable*)newObjectFromPool<HashTable, IMemPool*, int>(fMemPool, fMemPool, htsize);
    stable = (StringTable*)newObjectFromPool<StringTable, IMemPool*>(fMemPool, fMemPool);
    amap            = nullptr;
    arev            = nullptr;
    nextFree = 0;
    size = 0;
    caseInsensitive = nullptr;
    ciSize = 0;
    memPool = fMemPool;

    GrowAtomTable(INIT_ATOM_TABLE_SIZE);

    // Initialize lower part of atom table to "<undefined>" atom:

    AddAtomFixed("<undefined>", 0);
} // AtomTable::AtomTable

/*
 * AtomTable::Clone() -- clone an atomtable into a new IMemPool
 */
AtomTable *AtomTable::Clone(IMemPool *fMemPool) const
{
    AtomTable *newTable = (AtomTable*)newObjectFromPool<AtomTable, IMemPool*>(fMemPool, fMemPool);
    newTable->ops = &AtomTable_ops;
    newTable->memPool = fMemPool;
    newTable->stable = stable->Clone(fMemPool);
    newTable->htable = htable->Clone(fMemPool);
    newTable->nextFree = nextFree;
    newTable->size = size;
    newTable->ciSize = ciSize;
    if (size) {
        newTable->amap = (int*)newFromPool(fMemPool, sizeof(int)*size); // new int[size];
        newTable->arev = (int*)newFromPool(fMemPool, sizeof(int)*size); // new int[size];
        memcpy(newTable->amap, amap, size*sizeof(int));
        memcpy(newTable->arev, arev, size*sizeof(int));
    } else {
        newTable->amap = nullptr;
        newTable->arev = nullptr;
    }
    if (ciSize) {
        newTable->caseInsensitive = (char*)newFromPool(fMemPool, sizeof(char)*ciSize); // new char[ciSize];
        memcpy(newTable->caseInsensitive, caseInsensitive, ciSize*sizeof(int));
    } else {
        newTable->caseInsensitive = nullptr;
    }
    return newTable;
} // AtomTable::Clone

/*
 * AtomTable::GrowAtomTable() - Grow the atom table to at least "size" if it's smaller.
 *
 */

int AtomTable::GrowAtomTable(int fsize)
{
    int *newmap, *newrev, ii;

    if (size < fsize) {
        newmap = (int*)newFromPool(memPool, sizeof(int)*fsize); // new int[fsize];
        newrev = (int*)newFromPool(memPool, sizeof(int)*fsize); // new int[fsize];
        if (amap) {
            for (ii = 0; ii < size; ii++) {
                newmap[ii] = amap[ii];
                newrev[ii] = arev[ii];
            }
        } else {
            size = 0;
        }
        for (ii = size; ii < fsize; ii++) {
            newmap[ii] = 0;
            newrev[ii] = 0;
        }
        //delete(memPool) [] amap;
        //delete(memPool) [] arev; 
        amap = newmap;
        arev = newrev;
        size = fsize;
    }
    return 0;
} // AtomTable::GrowAtomTable

/*
 * lReverse() - Reverse the bottom 20 bits of a 32 bit int.
 *
 */

static int lReverse(int fval)
{
    unsigned int in = fval;
    int result = 0, cnt = 0;

    while(in) {
        result <<= 1;
        result |= in&1;
        in >>= 1;
        cnt++;
    }

    // Don't use all 31 bits.  One million atoms is plenty and sometimes the
    // upper bits are used for other things.

    if (cnt < 20)
        result <<= 20 - cnt;
    return result;
} // lReverse

/*
 * AtomTable::lAllocateAtom() - Allocate a new atom.  Associated with the "undefined" value of -1.
 *
 */

int AtomTable::lAllocateAtom(void)
{
    if (!nextFree && !size) {
        // We might get here if the constructor failed allocation
        // previously. Try again; if this doesn't work, we'll throw
        // an exception.
        GrowAtomTable(INIT_ATOM_TABLE_SIZE);
    } else if (nextFree >= size) {
        GrowAtomTable(nextFree*2);
    }
    amap[nextFree] = -1;
    arev[nextFree] = lReverse(nextFree);
    nextFree++;
    return nextFree - 1;
} // AtomTable::lAllocateAtom

/*
 * AtomTable::lSetAtomValue() - Allocate a new atom associated with "hashindex".
 *
 */

void AtomTable::lSetAtomValue(int atomNumber, int hashIndex)
{
    amap[atomNumber] = htable->entry[hashIndex].index;
    htable->entry[hashIndex].value = atomNumber;
} // AtomTable::lSetAtomValue

/*
 * AtomTable::lFindHashLoc() - Find the hash location for this string.  If fHTable is not NULL,
 *         find the location in that table, otherwise use the one in this atom table.
 *         Return -1 if the hash table is full.
 */

int AtomTable::lFindHashLoc(const char *s, HashTable *fHTable)
{
    int hashloc, hashdelta, count;
    int FoundEmptySlot = 0;
    //int collision[HASH_TABLE_MAX_COLLISIONS + 1];

    if( fHTable == nullptr )
        fHTable = htable;
    hashloc = HashString(s) % fHTable->size;
    if (!Empty(fHTable, hashloc)) {
        if (Match(fHTable, stable, s, hashloc))
            return hashloc;
        //collision[0] = hashloc;
        hashdelta = HashString2(s);
        count = 0;
        while (count < HASH_TABLE_MAX_COLLISIONS) {
            hashloc = ((hashloc + hashdelta) & 0x7fffffff) % fHTable->size;
            if (!Empty(fHTable, hashloc)) {
                if (Match(fHTable, stable, s, hashloc)) {
                    return hashloc;
                }
            } else {
                FoundEmptySlot = 1;
                break;
            }
            count++;
            //collision[count] = hashloc;
        }

        if (!FoundEmptySlot) {
#if 000
            if (Cg->options.DumpAtomTable) {
                int ii;
                TRACE_LEVEL(TRACELEVEL, ("*** Hash faild with more than %d collisions.  Must increase hash table size. ***\n",
                       HASH_TABLE_MAX_COLLISIONS));
                TRACE_LEVEL(TRACELEVEL, ("*** New string \"%s\", hash=%04x, delta=%04x.\n", s, collision[0], hashdelta));
                for (ii = 0; ii <= HASH_TABLE_MAX_COLLISIONS; ii++)
                    TRACE_LEVEL(TRACELEVEL, ("*** Collides on try %d at hash entry %04x with \"%s\"\n",
                           ii + 1, collision[ii], GetAtomString(fHTable->entry[collision[ii]].value)));
            }
#endif
            return -1;
        } else {
            fHTable->counts[count]++;
        }
    }
    return hashloc;
} // AtomTable::lFindHashLoc

/*
 * AtomTable::lIncreaseHashTableSize() - Increase the size of an atom table's hash table.
 *
 */

void AtomTable::lIncreaseHashTableSize(void)
{
    int newSize, ii;
    // HashTable *oldHashTable;

    // Save pointer to old hash table and allocate a new, larger one:

    // oldHashTable = htable;
    newSize = htable->size*2 + 1;
    htable = (HashTable*)newObjectFromPool<HashTable, IMemPool*, int>(memPool, memPool, newSize); // new HashTable(memPool, newSize);

    // Add all the existing values to the new atom table preserving their atom values:

    for (ii = 0; ii < nextFree; ii++)
    {
        const int offset = amap[ii];
        if (offset>0)  // -1 is undefined, 0 is empty
          AddAtomFixed(&stable->strings[offset], ii);
    }

    //delete(memPool) oldHashTable;
} // AtomTable::lIncreaseHashTableSize

/*
 * AtomTable::lLookUpAddStringHash() - Lookup a string in the hash table.  If it's not there,
 *        add it and initialize the atom value in the hash table to 0.
 *
 * Return the hash table index.
 *
 */

int AtomTable::lLookUpAddStringHash(const char *fStr)
{
    int hashloc, strloc;

    while(true) {
        hashloc = lFindHashLoc(fStr);
        if (hashloc >= 0)
            break;
        lIncreaseHashTableSize();
    }

    if (Empty(htable, hashloc)) {
        htable->entries++;
        strloc = AddString(stable, fStr);
        htable->entry[hashloc].index = strloc;
        htable->entry[hashloc].value = 0;
    }
    return hashloc;
} // AtomTable::lLookUpAddStringHash

/*
 * AtomTable::lCheckCaseInsensitiveAtom() - Is fStr a case insensitive value in this table.
 *         Note: Returns FALSE for all strings longer than 31 characters.
 */

int AtomTable::lCheckCaseInsensitiveAtom(const char *fStr)
{
    int hashloc, atom, len;
    char *p, lBuff[32];

    len = (int) strlen(fStr);
    if (len >= (int) sizeof(lBuff))
        return 0;
    p = lBuff;
    while (*fStr) {
        if (*fStr >= 'A' && *fStr <= 'Z')
            *p++ = 'a' + (*fStr - 'A');
        else
            *p++ = *fStr;
        fStr++;
    }
    *p = 0;
    hashloc = lFindHashLoc(lBuff);
    if (hashloc < 0 || Empty(htable, hashloc))
        return 0;
    atom = htable->entry[hashloc].value;
    if (atom > 0 && atom < ciSize && caseInsensitive[atom])
        return atom;
    return 0;
} // AtomTable::lCheckCaseInsensitiveAtom

/*
 * AtomTable::LookUpAddString() - Lookup a string in the hash table.  If it's not there, add
 *        it and initialize the atom value in the hash table to the next atom number.
 *        Return the atom value of string.
 */

/// !!!! XYZZY Replace LookUpAddString() with AddAtom()!!!
/// !!!! XYZZY Replace LookUpAddString() with AddAtom()!!!
/// !!!! XYZZY Replace LookUpAddString() with AddAtom()!!!
/// !!!! XYZZY Replace LookUpAddString() with AddAtom()!!!

int AtomTable::LookUpAddString(const char *fStr)
{
    int hashIndex, atom;

    hashIndex = lLookUpAddStringHash(fStr);
    atom = htable->entry[hashIndex].value;
    if (atom == 0) {
        atom = lCheckCaseInsensitiveAtom(fStr);
        if (atom == 0) {
            atom = lAllocateAtom();
        }
        lSetAtomValue(atom, hashIndex);
    }
    return atom;
} // AtomTable::LookUpAddString

int AtomTable::LookUpString(const char *fStr)
{
    int hashIndex, atom = 0;

    hashIndex = lFindHashLoc(fStr);
    if (hashIndex >= 0 && !Empty(htable, hashIndex))
        atom = htable->entry[hashIndex].value;
    if (atom == 0) {
        atom = lCheckCaseInsensitiveAtom(fStr);
    }
    return atom;
} // AtomTable::LookUpAddString

/*
 * AtomTable::GetAtomString() - Return a string representation of an atom.
 *
 */

const char *AtomTable::GetAtomString(int atom)
{
    int soffset;
    static char ilwalidBuffer[4][32];
    static int ilwalidBufferIdx = 0;

    if (atom > 0 && atom < nextFree) {
        soffset = amap[atom];
        if (soffset > 0 && soffset < stable->nextFree) {
            return &stable->strings[soffset];
        } else {
            return "<internal error: bad soffset>";
        }
    } else {
        if (atom == 0) {
            return "<null atom>";
        } else {
            /* not thread safe, but only needed for debugging. */
            ilwalidBufferIdx = (ilwalidBufferIdx + 1) & 3;
            sprintf(ilwalidBuffer[ilwalidBufferIdx], "<invalid atom %d>", atom);
            return ilwalidBuffer[ilwalidBufferIdx];
        }
    }
} // AtomTable::GetAtomString

/*
 * GetReversedAtom()
 *
 */

int AtomTable::GetReversedAtom(int atom)
{
    if (atom > 0 && atom < nextFree) {
        return arev[atom];
    } else {
        return 0;
    }
} // AtomTable::GetReversedAtom

/*
 * AtomTable::AddAtom() - Add a string to the atom, hash and string tables if it isn't already
 *         there.  Return it's atom index.
 */

int AtomTable::AddAtom(const char *fStr)
{
    return LookUpAddString(fStr);
} // AtomTable::AddAtom

/*
 * AtomTable::AddAtomFixed() - Add an atom to a fixed position in the hash and string tables if
 *         it isn't already there. Assign it the atom value of "atom".
 */

int AtomTable::AddAtomFixed(const char *fStr, int atom)
{
    int hashIndex, lSize;

    hashIndex = lLookUpAddStringHash(fStr);
    if (nextFree >= size || atom >= size) {
        lSize = size*2;
        if (lSize <= atom)
            lSize = atom + 1;
        GrowAtomTable(lSize);
    }
    amap[atom] = htable->entry[hashIndex].index;
    htable->entry[hashIndex].value = atom;
    //if (atom >= atable->nextFree)
    //    atable->nextFree = atom + 1;
    while (atom >= nextFree) {
        arev[nextFree] = lReverse(nextFree);
        nextFree++;
    }
    return atom;
} // AtomTable::AddAtomFixed

/*
 * AtomTable::MakeCaseInsensitive() -- make an atom case insensitive.
 *  THE ATOM MUST BE ALL LOWERCASE
 */
void AtomTable::MakeCaseInsensitive(int atom)
{
    if (atom >= ciSize) {
        char *old = caseInsensitive;
        caseInsensitive = new char[atom + 1];
        if (ciSize > 0) {
            memcpy(caseInsensitive, old, ciSize);
        }
        memset(caseInsensitive + ciSize, 0, atom - ciSize);
        ciSize = atom+1;
        // free(old);
    }
    caseInsensitive[atom] = 1;
} // AtomTable::MakeCaseInsensitive

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Interface Functions: /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/*
 * NewIAtomTable() - Allocate a new AtomTable and return an opaque pointer to it.
 *
 */

IAtomTable *NewIAtomTable(IMemPool *fMemPool, int htsize)
{
    AtomTable *atable;

    atable = (AtomTable*)newObjectFromPool<AtomTable, IMemPool*, int>(fMemPool, fMemPool, htsize); // new AtomTable(fMemPool, htsize);
    return atable;
} // NewIAtomTable

#if 00
/*
 * CloneIAtomTable() - Allocate a new AtomTable that is a copy on an existing
 *  table and return an opaque pointer to it.
 *
 */

IAtomTable *CloneIAtomTable(IAtomTable *atable, IMemPool *fMemPool)
{
    return static_cast<AtomTable *>(atable)->Clone(fMemPool);
} // CloneIAtomTable
#endif

/*
 * AddIAtomFixed() - Define an atom with a fixed value.  Overwrites existing string for that
 *        atom if one is already defined.
 */

int AddIAtomFixed(IAtomTable *atable, const char *fStr, int atom)
{
    return static_cast<AtomTable *>(atable)->AddAtomFixed(fStr, atom);
} // AddIAtomFixed

#if 00
/*
 * MakeIAtomCaseInsensitive() -- make an atom case insensitive -- atom must
 *      be all lowercase
 */
void MakeIAtomCaseInsensitive(IAtomTable *atable, int atom)
{
    static_cast<AtomTable *>(atable)->MakeCaseInsensitive(atom);
} // MakeIAtomCaseInsenstive
#endif

#if defined(EXPORTSYMBOLS)
#if defined(WIN32)
#define DLLEXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__>=4
#define DLLEXPORT __attribute__ ((visibility("default")))
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define DLLEXPORT __global
#else
#define DLLEXPORT
#endif
#else
#define DLLEXPORT
#endif

#ifdef GetIAtomString
#undef GetIAtomString
extern "C" {
/* for backwards compatability with older programs built with the old
 * interface file, we need to provide this as a dll export. */
/*
 * GetIAtomString() - Return a pointer to the string value of an atom.
 *
 */

DLLEXPORT const char *GetIAtomString(IAtomTable *atable, int atom) {
    return atable->ops->getString(atable, atom);
} // GetIAtomString
}
#endif

#undef DLLEXPORT

/*
 * GetReversedIAtom() - Return the reversed binary representation of an atom.
 *
 */

int GetReversedIAtom(IAtomTable *atable, int atom)
{
    return static_cast<AtomTable *>(atable)->GetReversedAtom(atom);
} // GetReversedIAtom

/*
 * FreeIAtomTable() - Free and atom table.
 *
 */

void FreeIAtomTable(IAtomTable *atable)
{
    AtomTable *ptr = static_cast<AtomTable *>(atable);
    ptr->amap      = nullptr;
    ptr->arev      = nullptr;
    ptr->nextFree = 0;
    ptr->size = 0;
} // FreeIAtomTable


void initializeAtomTable(IMemPool& memPool, const std::string& descr, IAtomTable **atomTable)
{
  // State machine's global memory atom table
  memPool.bytes = 0;
  memPool.MemAlloc    = (void* (*)(void*, size_t))memspMalloc;
  memPool.MemFree    = (void (*)(void*, void*))memspFree;
  memPool.memAllocArg = memspCreate( descr.c_str(), stdLwrrentMemspace, 0x1000 );
  *atomTable = NewIAtomTable(&memPool, 0);
}
