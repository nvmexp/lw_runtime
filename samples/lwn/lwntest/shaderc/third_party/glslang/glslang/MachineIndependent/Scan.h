//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
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
#ifndef _GLSLANG_SCAN_INCLUDED_
#define _GLSLANG_SCAN_INCLUDED_

#include "Versions.h"

namespace glslang {

// Use a global end-of-input character, so no translation is needed across
// layers of encapsulation.  Characters are all 8 bit, and positive, so there is
// no aliasing of character 255 onto -1, for example.
const int EndOfInput = -1;

//
// A character scanner that seamlessly, on read-only strings, reads across an
// array of strings without assuming null termination.
//
class TInputScanner {
public:
    TInputScanner(int n, const char* const s[], size_t L[], const char* const* names = nullptr,
                  int b = 0, int f = 0, bool single = false) :
        numSources(n),
         // up to this point, common usage is "char*", but now we need positive 8-bit characters
        sources(reinterpret_cast<const unsigned char* const *>(s)),
        lengths(L), lwrrentSource(0), lwrrentChar(0), stringBias(b), finale(f), singleLogical(single),
        endOfFileReached(false)
    {
        loc = new TSourceLoc[numSources];
        for (int i = 0; i < numSources; ++i) {
            loc[i].init(i - stringBias);
        }
        if (names != nullptr) {
            for (int i = 0; i < numSources; ++i)
                loc[i].name = names[i] != nullptr ? NewPoolTString(names[i]) : nullptr;
        }
        loc[lwrrentSource].line = 1;
        logicalSourceLoc.init(1);
        logicalSourceLoc.name = loc[0].name;
    }

    virtual ~TInputScanner()
    {
        delete [] loc;
    }

    // retrieve the next character and advance one character
    int get()
    {
        int ret = peek();
        if (ret == EndOfInput)
            return ret;
        ++loc[lwrrentSource].column;
        ++logicalSourceLoc.column;
        if (ret == '\n') {
            ++loc[lwrrentSource].line;
            ++logicalSourceLoc.line;
            logicalSourceLoc.column = 0;
            loc[lwrrentSource].column = 0;
        }
        advance();

        return ret;
    }

    // retrieve the next character, no advance
    int peek()
    {
        if (lwrrentSource >= numSources) {
            endOfFileReached = true;
            return EndOfInput;
        }
        // Make sure we do not read off the end of a string.
        // N.B. Sources can have a length of 0.
        int sourceToRead = lwrrentSource;
        size_t charToRead = lwrrentChar;
        while(charToRead >= lengths[sourceToRead]) {
            charToRead = 0;
            sourceToRead += 1;
            if (sourceToRead >= numSources) {
                return EndOfInput;
            }
        }

        // Here, we care about making negative valued characters positive
        return sources[sourceToRead][charToRead];
    }

    // go back one character
    void unget()
    {
        // Do not roll back once we've reached the end of the file.
        if (endOfFileReached)
            return;

        if (lwrrentChar > 0) {
            --lwrrentChar;
            --loc[lwrrentSource].column;
            --logicalSourceLoc.column;
            if (loc[lwrrentSource].column < 0) {
                // We've moved back past a new line. Find the
                // previous newline (or start of the file) to compute
                // the column count on the now current line.
                size_t chIndex = lwrrentChar;
                while (chIndex > 0) {
                    if (sources[lwrrentSource][chIndex] == '\n') {
                        break;
                    }
                    --chIndex;
                }
                logicalSourceLoc.column = (int)(lwrrentChar - chIndex);
                loc[lwrrentSource].column = (int)(lwrrentChar - chIndex);
            }
        } else {
            do {
                --lwrrentSource;
            } while (lwrrentSource > 0 && lengths[lwrrentSource] == 0);
            if (lengths[lwrrentSource] == 0) {
                // set to 0 if we've backed up to the start of an empty string
                lwrrentChar = 0;
            } else
                lwrrentChar = lengths[lwrrentSource] - 1;
        }
        if (peek() == '\n') {
            --loc[lwrrentSource].line;
            --logicalSourceLoc.line;
        }
    }

    // for #line override
    void setLine(int newLine)
    {
        logicalSourceLoc.line = newLine;
        loc[getLastValidSourceIndex()].line = newLine;
    }

    // for #line override in filename based parsing
    void setFile(const char* filename)
    {
        TString* fn_tstr = NewPoolTString(filename);
        logicalSourceLoc.name = fn_tstr;
        loc[getLastValidSourceIndex()].name = fn_tstr;
    }

    void setFile(const char* filename, int i)
    {
        TString* fn_tstr = NewPoolTString(filename);
        if (i == getLastValidSourceIndex()) {
            logicalSourceLoc.name = fn_tstr;
        }
        loc[i].name = fn_tstr;
    }

    void setString(int newString)
    {
        logicalSourceLoc.string = newString;
        loc[getLastValidSourceIndex()].string = newString;
        logicalSourceLoc.name = nullptr;
        loc[getLastValidSourceIndex()].name = nullptr;
    }

    // for #include content indentation
    void setColumn(int col)
    {
        logicalSourceLoc.column = col;
        loc[getLastValidSourceIndex()].column = col;
    }

    void setEndOfInput()
    {
        endOfFileReached = true;
        lwrrentSource = numSources;
    }

    bool atEndOfInput() const { return endOfFileReached; }

    const TSourceLoc& getSourceLoc() const
    {
        if (singleLogical) {
            return logicalSourceLoc;
        } else {
            return loc[std::max(0, std::min(lwrrentSource, numSources - finale - 1))];
        }
    }
    // Returns the index (starting from 0) of the most recent valid source string we are reading from.
    int getLastValidSourceIndex() const { return std::min(lwrrentSource, numSources - 1); }

    void consumeWhiteSpace(bool& foundNonSpaceTab);
    bool consumeComment();
    void consumeWhitespaceComment(bool& foundNonSpaceTab);
    bool scalwersion(int& version, EProfile& profile, bool& notFirstToken);

protected:

    // advance one character
    void advance()
    {
        ++lwrrentChar;
        if (lwrrentChar >= lengths[lwrrentSource]) {
            ++lwrrentSource;
            if (lwrrentSource < numSources) {
                loc[lwrrentSource].string = loc[lwrrentSource - 1].string + 1;
                loc[lwrrentSource].line = 1;
                loc[lwrrentSource].column = 0;
            }
            while (lwrrentSource < numSources && lengths[lwrrentSource] == 0) {
                ++lwrrentSource;
                if (lwrrentSource < numSources) {
                    loc[lwrrentSource].string = loc[lwrrentSource - 1].string + 1;
                    loc[lwrrentSource].line = 1;
                    loc[lwrrentSource].column = 0;
                }
            }
            lwrrentChar = 0;
        }
    }

    int numSources;                      // number of strings in source
    const unsigned char* const *sources; // array of strings; must be colwerted to positive values on use, to avoid aliasing with -1 as EndOfInput
    const size_t *lengths;               // length of each string
    int lwrrentSource;
    size_t lwrrentChar;

    // This is for reporting what string/line an error oclwrred on, and can be overridden by #line.
    // It remembers the last state of each source string as it is left for the next one, so unget()
    // can restore that state.
    TSourceLoc* loc;  // an array

    int stringBias;   // the first string that is the user's string number 0
    int finale;       // number of internal strings after user's last string

    TSourceLoc logicalSourceLoc;
    bool singleLogical; // treats the strings as a single logical string.
                        // locations will be reported from the first string.

    // Set to true once peek() returns EndOfFile, so that we won't roll back
    // once we've reached EndOfFile.
    bool endOfFileReached;
};

} // end namespace glslang

#endif // _GLSLANG_SCAN_INCLUDED_
