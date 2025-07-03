/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/****************************** LwWatch ***********************************\
*                                                                          *
*                             HDCP routines                                *
*                                                                          *
\***************************************************************************/

#include "lwwatch.h"
#include "inc/disp.h"
#include "exts.h"

//
// Display the HDCP releated help info
//
void
hdcpDisplayHelp(void)
{
    dprintf("HDCP commands:\n");
    dprintf(" \"hdcp --help\"                    - Displays the HDCP related help menu\n");
    dprintf(" \"hdcp status\"                    - Displays the HDCP OR status\n");
    dprintf(" \"hdcp status SOR [0-7]\" or, \n");
    dprintf(" \"hdcp status SOR *\"              - Displays the HDCP Register info for particular/all attached SOR\n");
    dprintf(" \"hdcp keydecryption status\"      - Displays the HDCP key decryption status\n");
    dprintf(" \"hdcp keydecryption trigger\"     - Trigger the HDCP key decryption and displays the key decryption status\n\n");
}