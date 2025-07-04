#!/bin/bash
#
# Copyright 2008, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Exelwtes the samples and tests for the Google Test Framework.

# Help the dynamic linker find the path to the libraries.
export DYLD_FRAMEWORK_PATH=$BUILT_PRODUCTS_DIR
export DYLD_LIBRARY_PATH=$BUILT_PRODUCTS_DIR

# Create some exelwtables.
test_exelwtables=("$BUILT_PRODUCTS_DIR/gtest_unittest-framework"
                  "$BUILT_PRODUCTS_DIR/gtest_unittest"
                  "$BUILT_PRODUCTS_DIR/sample1_unittest-framework"
                  "$BUILT_PRODUCTS_DIR/sample1_unittest-static")

# Now execute each one in turn keeping track of how many succeeded and failed. 
succeeded=0
failed=0
failed_list=()
for test in ${test_exelwtables[*]}; do
  "$test"
  result=$?
  if [ $result -eq 0 ]; then
    succeeded=$(( $succeeded + 1 ))
  else
    failed=$(( failed + 1 ))
    failed_list="$failed_list $test"
  fi
done

# Report the successes and failures to the console.
echo "Tests complete with $succeeded successes and $failed failures."
if [ $failed -ne 0 ]; then
  echo "The following tests failed:"
  echo $failed_list
fi
exit $failed
