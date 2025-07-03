/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! includes, system
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <map>

// includes, projcet
#include "matlab.h"

// namespace, unnamed
namespace {}  // end namespace, unnamed

///////////////////////////////////////////////////////////////////////////////
//! Write a tridiagonal, symmetric matrix in vector representation and
//! it's eigelwalues
//! @param  filename  name of output file
//! @param  d  diagonal entries of the matrix
//! @param  s  superdiagonal entries of the matrix (len = n - 1)
//! @param  eigelwals  eigelwalues of the matrix
//! @param  indices  vector of len n containing the position of the eigelwalues
//!                  if these are sorted in ascending order
//! @param  n  size of the matrix
///////////////////////////////////////////////////////////////////////////////
void writeTridiagSymMatlab(const char *filename, float *d, float *s,
                           float *eigelwals, const unsigned int n) {
  std::ofstream file(filename, std::ios::out);

  // write diagonal entries
  writeVectorMatlab(file, "d", d, n);

  // write superdiagonal entries
  writeVectorMatlab(file, "s", s, n - 1);

  // write eigelwalues
  writeVectorMatlab(file, "eigvals", eigelwals, n);

  file.close();
}
