//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
/****************************************************************************\
Copyright (c) 2002, LWPU Corporation.

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
// For recording and playing back the stream of tokens in a macro definition.
//

#ifndef _CRT_SELWRE_NO_WARNINGS
#define _CRT_SELWRE_NO_WARNINGS
#endif
#if (defined(_MSC_VER) && _MSC_VER < 1900 /*vs2015*/)
#define snprintf sprintf_s
#endif

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cctype>

#include "PpContext.h"
#include "PpTokens.h"

namespace glslang {

// Add a token (including backing string) to the end of a macro
// token stream, for later playback.
void TPpContext::TokenStream::putToken(int atom, TPpToken* ppToken)
{
    TokenStream::Token streamToken(atom, *ppToken);
    stream.push_back(streamToken);
}

// Read the next token from a macro token stream.
int TPpContext::TokenStream::getToken(TParseContextBase& parseContext, TPpToken *ppToken)
{
    if (atEnd())
        return EndOfInput;

    int atom = stream[lwrrentPos++].get(*ppToken);
    ppToken->loc = parseContext.getLwrrentLoc();

#ifndef GLSLANG_WEB
    // Check for ##, unless the current # is the last character
    if (atom == '#') {
        if (peekToken('#')) {
            parseContext.requireProfile(ppToken->loc, ~EEsProfile, "token pasting (##)");
            parseContext.profileRequires(ppToken->loc, ~EEsProfile, 130, 0, "token pasting (##)");
            lwrrentPos++;
            atom = PpAtomPaste;
        }
    }
#endif

    return atom;
}

// We are pasting if
//   1. we are preceding a pasting operator within this stream
// or
//   2. the entire macro is preceding a pasting operator (lastTokenPastes)
//      and we are also on the last token
bool TPpContext::TokenStream::peekTokenizedPasting(bool lastTokenPastes)
{
    // 1. preceding ##?

    size_t savePos = lwrrentPos;
    // skip white space
    while (peekToken(' '))
        ++lwrrentPos;
    if (peekToken(PpAtomPaste)) {
        lwrrentPos = savePos;
        return true;
    }

    // 2. last token and we've been told after this there will be a ##

    if (! lastTokenPastes)
        return false;
    // Getting here means the last token will be pasted, after this

    // Are we at the last non-whitespace token?
    savePos = lwrrentPos;
    bool moreTokens = false;
    do {
        if (atEnd())
            break;
        if (!peekToken(' ')) {
            moreTokens = true;
            break;
        }
        ++lwrrentPos;
    } while (true);
    lwrrentPos = savePos;

    return !moreTokens;
}

// See if the next non-white-space tokens are two conselwtive #
bool TPpContext::TokenStream::peekUntokenizedPasting()
{
    // don't return early, have to restore this
    size_t savePos = lwrrentPos;

    // skip white-space
    while (peekToken(' '))
        ++lwrrentPos;

    // check for ##
    bool pasting = false;
    if (peekToken('#')) {
        ++lwrrentPos;
        if (peekToken('#'))
            pasting = true;
    }

    lwrrentPos = savePos;

    return pasting;
}

void TPpContext::pushTokenStreamInput(TokenStream& ts, bool prepasting)
{
    pushInput(new tTokenInput(this, &ts, prepasting));
    ts.reset();
}

int TPpContext::tUngotTokenInput::scan(TPpToken* ppToken)
{
    if (done)
        return EndOfInput;

    int ret = token;
    *ppToken = lval;
    done = true;

    return ret;
}

void TPpContext::UngetToken(int token, TPpToken* ppToken)
{
    pushInput(new tUngotTokenInput(this, token, ppToken));
}

} // end namespace glslang
