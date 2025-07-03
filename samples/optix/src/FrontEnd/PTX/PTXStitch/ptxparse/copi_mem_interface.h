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
// mem_interface.h - Memory allocation interface header.
//

#if !defined(__MEM_INTERFACE_H)
#define __MEM_INTERFACE_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // For size_t
#include "lwtypes.h" // for LwU64

typedef struct IMemPool_rec {
    void *(*MemAlloc)(void *, size_t);          // Memory allocation callback
    void (*MemFree)(void *, void *);            // Memory free callback
    void *memAllocArg;                          // Param for MemAlloc
    LwU64 bytes;
} IMemPool;

#ifndef CG_NOGLOBAL_ALLOC

#ifdef __cplusplus
void *operator new(size_t size, struct IMemPool_rec *fMemPool);
void *operator new[](size_t size, struct IMemPool_rec *fMemPool);
void operator delete(void *p, struct IMemPool_rec *fMemPool);
void operator delete[](void *p, struct IMemPool_rec *fMemPool);
#endif

#endif

void InitIMemPool(IMemPool *fMemPool);

#ifdef __cplusplus
}
#endif

#endif // !defined(__MEM_INTERFACE_H)
