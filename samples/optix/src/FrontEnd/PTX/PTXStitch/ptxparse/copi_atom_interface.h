/****************************************************************************\
Copyright (c) 2017, LWPU CORPORATION.

LWPU Corporation("LWPU") supplies this software to you in
consideration of your agreement to the following terms, and your use,
installation, modification or redistribution of this LWPU software
constitutes acceptance of these terms.  If you do not agree with these
terms, please do not use, install, modify or redistribute this LWPU
software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, LWPU grants you a personal, non-exclusive
license, under LWPU's copyrights in this original LWPU software (the
"LWPU Software"), to use, reproduce, modify and redistribute the
LWPU Software, with or without modifications, in source and/or binary
forms; provided that if you redistribute the LWPU Software, you must
retain the copyright notice of LWPU, this notice and the following
text and disclaimers in all such redistributions of the LWPU Software.
Neither the name, trademarks, service marks nor logos of LWPU
Corporation may be used to endorse or promote products derived from the
LWPU Software without specific prior written permission from LWPU.
Except as expressly stated in this notice, no other rights or licenses
express or implied, are granted by LWPU herein, including but not
limited to any patent rights that may be infringed by your derivative
works or by other works in which the LWPU Software may be
incorporated. No hardware is licensed hereunder. 

THE LWPU SOFTWARE IS BEING PROVIDED ON AN "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING WITHOUT LIMITATION, WARRANTIES OR CONDITIONS OF TITLE,
NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
ITS USE AND OPERATION EITHER ALONE OR IN COMBINATION WITH OTHER
PRODUCTS.

IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT,
INCIDENTAL, EXEMPLARY, CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, LOST PROFITS; PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) OR ARISING IN ANY WAY
OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION OF THE
LWPU SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT,
TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF
LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************/

//
// atom_interface.h
//

#if !defined(__ATOM_INTERFACE_H)
#define __ATOM_INTERFACE_H 1

#include "copi_mem_interface.h"

#if defined(EXPORTSYMBOLS)
#define GetIAtomString inline_GetIAtomString
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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IAtomTable_Rec IAtomTable;

struct IAtomTable_ops {
    int (*addAtom)(IAtomTable *atable, const char *fStr);
    const char *(*getString)(IAtomTable *atable, int atom);
    int (*lookupAtom)(IAtomTable *atable, const char *fStr); // lookup without adding
};

struct IAtomTable_Rec {
    struct IAtomTable_ops       *ops;
};

IAtomTable *NewIAtomTable(IMemPool *fMemPool, int htsize);

int AddIAtomFixed(IAtomTable *atable, const char *fStr, int atom);
static LW_INLINE int AddIAtom(IAtomTable *atable, const char *fStr) {
    return atable->ops->addAtom(atable, fStr);
}
static LW_INLINE int LookUpAddIString(IAtomTable *atable, const char *fStr) {
    return atable->ops->addAtom(atable, fStr);
}
static LW_INLINE int LookUpIString(IAtomTable *atable, const char *fStr) {
    return atable->ops->lookupAtom(atable, fStr);
}
static LW_INLINE const char *GetIAtomString(IAtomTable *atable, int atom) {
    return atable->ops->getString(atable, atom);
}
int GetReversedIAtom(IAtomTable *atable, int atom);

#if 00
/* none of these are used by anyone */
IAtomTable *CloneIAtomTable(IAtomTable *atable, IMemPool *fMemPool);
void MakeIAtomCaseInsensitive(IAtomTable *atable, int atom);
int LowerIAtom(IAtomTable *atable, int atom);
int LookUpAddIAtomString(IAtomTable *atable, const char *fStr);

//const char *Atom2String(int atom);
#endif

void FreeIAtomTable(IAtomTable *atable);
void PrintIAtomTable(IAtomTable *atable);

#ifdef __cplusplus
}
#endif

#undef DLLEXPORT

#endif // !defined(__ATOM_INTERFACE_H)
