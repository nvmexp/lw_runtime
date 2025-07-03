// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <string>
#include <vector>


namespace prodlib {
/* \brief Create a backtrace of the calling thread. Each string in the returned vector contains information about one line
     * \param firstFrame Skip the given amount of frames in the backtrace.
     * \param maxFrames Walk up the stack up to of maxFrames after skipping skipFrames.
     * \param exclusionList optional list of names to exclude in the backtrace
     * \return A std::vector containing strings where each string contains a textual representation for one stackframe.
     */
std::vector<std::string> backtrace( int                             skipFrames,
                                    unsigned int                    maxFrames,
                                    const std::vector<std::string>& exclusionList = std::vector<std::string>() );
}  // end namespace prodlib
