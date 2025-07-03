#ifndef __LWW_PDB_H
#define __LWW_PDB_H

#include "os.h"

char * pdbEnumFromOdbClass(
    const char* odbClassName);

void pdbDump(
    ULONG64     rootObjAddr, 
    const char* propEnumName);

#endif __LWW_PDB_H