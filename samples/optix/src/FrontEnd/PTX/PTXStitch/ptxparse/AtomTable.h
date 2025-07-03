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

#include <copi_mem_interface.h>
#include <copi_atom_interface.h>
#include <prodlib/exceptions/Assert.h>
#include <string.h>

  const int INIT_STRING_TABLE_SIZE = 16384;

  class StringTable {
  public:
    StringTable(IMemPool *fMemPool, int size = INIT_STRING_TABLE_SIZE);
    StringTable *Clone(IMemPool *fMemPool) const;
  private:
    StringTable() {} // private -- used by clone
  public:
    IMemPool *memPool;   // Memory allocation
    char *strings;
    int nextFree;
    int size;
  };

#define INIT_HASH_TABLE_SIZE 2047
#define HASH_TABLE_MAX_COLLISIONS 3

  class HashEntry {
  public:
    int index;      // String table offset of string representation
    int value;      // Atom (symbol) value
  };

  class HashTable {
  public:
    HashTable(IMemPool *fMemPool, int size = INIT_HASH_TABLE_SIZE);
    HashTable *Clone(IMemPool *fMemPool) const;
  public:
    IMemPool *memPool;   // Memory allocation
    HashEntry *entry;
    int size;
    int entries;
    int counts[HASH_TABLE_MAX_COLLISIONS + 1];
  };

#define INIT_ATOM_TABLE_SIZE 1024

  class AtomTable : public IAtomTable {
  public:
    AtomTable(IMemPool *fMem, int size = INIT_ATOM_TABLE_SIZE);
    int GrowAtomTable(int size);
  private:
    AtomTable() {} // private -- used by Clone
    int  lFindHashLoc( const char* fStr, HashTable* fHTable = nullptr );
    void lIncreaseHashTableSize(void);
    int lLookUpAddStringHash(const char *fStr);
    int lCheckCaseInsensitiveAtom(const char *fStr);
  public:
    int LookUpAddString(const char *fStr);
    int LookUpString(const char *fStr);
  private:
    int lAllocateAtom(void);
    void lSetAtomValue(int atomnumber, int hashindex);
  public:
    const char *GetAtomString(int atom);
    int GetReversedAtom(int atom);
    int AddAtom(const char *fStr);
    int AddAtomFixed(const char *fStr, int atom);
    void MakeCaseInsensitive(int atom);
    AtomTable *Clone(IMemPool *fMem) const;
  public:
    IMemPool *memPool;  // Memory allocation
    StringTable *stable;// String table.
    HashTable *htable;  // Hashes string to atom number and token value.
    // Multiple strings can have the same token value but
    // each unique string is a unique atom.
    int *amap;          // Maps atom value to offset in string table.  Atoms
    // all map to unique strings except for some undefined
    // values in the lower, fixed part of the atom table
    // that map to "<undefined>".  The lowest 256 atoms
    // correspond to single character ASCII values except
    // for alphanumeric characters and '_', which can be
    // other tokens.  Next come the language tokens with
    // their atom values equal to the token value.  Then
    // come predefined atoms, followed by user specified
    // identifiers.
    int *arev;          // Reversed atom for symbol table use.
    int nextFree;
    int size;
    char *caseInsensitive;     // identifies case insensitive atoms
    int ciSize;                // size of case_insensitive array
  };

  IAtomTable *NewIAtomTable(IMemPool *fMemPool, int htsize);
  int AddIAtomFixed(IAtomTable *atable, const char *fStr, int atom);
  int GetReversedIAtom(IAtomTable *atable, int atom);
  void FreeIAtomTable(IAtomTable *atable);
  void initializeAtomTable(IMemPool& memPool, const std::string& descr, IAtomTable **atomTable);

  /*
   * Memory allocator helpers
   *
   */
  static void* newFromPool(IMemPool *fMemPool, unsigned int size) // Equivalent: new char[size]
  {
    return fMemPool->MemAlloc(fMemPool->memAllocArg, size*sizeof(char));
  }
  template<typename classType> static void* newObjectFromPool(IMemPool *fMemPool) // Equivalent: new Object
  {
    void *ptr = (void*)fMemPool->MemAlloc(fMemPool->memAllocArg, sizeof(classType));
    classType obj;
    memcpy(ptr, &obj, sizeof(classType));
    return ptr;
  }
  template<typename classType, typename t0> static void* newObjectFromPool(IMemPool *fMemPool, t0 arg0) // Equivalent: new Object(arg0)
  {
    void *ptr = (void*)fMemPool->MemAlloc(fMemPool->memAllocArg, sizeof(classType));
    classType obj(arg0);
    memcpy(ptr, &obj, sizeof(classType));
    return ptr;
  }
  template<typename classType, typename t0, typename t1> static void* newObjectFromPool(IMemPool *fMemPool, t0 arg0, t1 arg1) // Equivalent: new Object(arg0, arg1)
  {
    void *ptr = (void*)fMemPool->MemAlloc(fMemPool->memAllocArg, sizeof(classType));
    classType obj(arg0, arg1);
    memcpy(ptr, &obj, sizeof(classType));
    return ptr;
  }

