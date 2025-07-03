/****************************************************************************\
Copyright (c) 2008-2016, LWPU Corporation.

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
// parseasm_profileoption.cpp
//

#include <string.h>
#include <stdlib.h>

#include "copi_inglobals.h"
#include "copi_dag_interface.h"
#include "parseasm_profileoption.h"
#if defined(AR20)
#include "copi_tegra_interface.h"
#include "parseasm_profileoption_ar20.h"
#endif //AR20
#include "parseasmdecls.h"

static void ProcessProfileOptions_base(const char *fBase, const char *fData, int HasNumber, 
                                       int fValue);

static void PrintHelp_base(void);

static void FillParamsStructure_base(void *This, LdParams *fLdParams);

static void CleanUp_base(void);

/*
 * InitializeProfileStruct()
 *
 */

void InitializeProfileStruct(void) {    

#if defined(AR20)
    if (o_Profile && !strcmp(o_Profile, "ar20fp")) {
        AllocateProfileStruct_ar20fp();
        InitializeProfileStruct_ar20fp();
    } else if (o_Profile && !strcmp(o_Profile, "ar20vp")) {
        AllocateProfileStruct_ar20vp();
        InitializeProfileStruct_ar20vp();
    } else 
#endif // AR20
    {
        o_ProfileStruct = (Parseasm_ProfileOption *) malloc(sizeof(Parseasm_ProfileOption));
        InitializeProfileStruct_base();
    }

} // InitializeProfileStruct

/*
 * FinalizeProfileStruct()
 *
 */

void FinalizeProfileStruct(void) {

    o_ProfileStruct->CleanUp();

    free(o_ProfileStruct);

} // FinalizeProfileStruct

/*
 * IntializeProfileStruct_base()
 *
 */

void InitializeProfileStruct_base(void)
{

    o_ProfileStruct->ProcessProfileOptions = ProcessProfileOptions_base;
    o_ProfileStruct->PrintHelp = PrintHelp_base;
    o_ProfileStruct->FillParamsStructure = FillParamsStructure_base;
    o_ProfileStruct->CleanUp = CleanUp_base;

} // InitializeProfileStruct_base

/*
 * ProcessProfileOptions_base()
 *
 */

static void ProcessProfileOptions_base(const char *fBase, const char *fData, int HasNumber, 
                                       int fValue)
{    

} // ProcessProfileOptions_base

/*
 * PrintHelp_base()
 *
 */

static void PrintHelp_base(void)
{

} // PrintHelp_base

/*
 * FillParamsStructure_base()
 *
 */

static void FillParamsStructure_base(void *This, LdParams *fLdParams)
{

} // FillParamsStructure_base

/*
 * CleanUp_base()
 *
 */

static void CleanUp_base(void)
{

} // CleanUp_base

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// End of parseasm_profileoption.cpp ////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
