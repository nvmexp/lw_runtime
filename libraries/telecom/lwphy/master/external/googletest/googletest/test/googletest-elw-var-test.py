#!/usr/bin/elw python
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

"""Verifies that Google Test correctly parses environment variables."""

import os
import gtest_test_utils


IS_WINDOWS = os.name == 'nt'
IS_LINUX = os.name == 'posix' and os.uname()[0] == 'Linux'

COMMAND = gtest_test_utils.GetTestExelwtablePath('googletest-elw-var-test_')

elwiron = os.elwiron.copy()


def AssertEq(expected, actual):
  if expected != actual:
    print 'Expected: %s' % (expected,)
    print '  Actual: %s' % (actual,)
    raise AssertionError


def SetElwVar(elw_var, value):
  """Sets the elw variable to 'value'; unsets it when 'value' is None."""

  if value is not None:
    elwiron[elw_var] = value
  elif elw_var in elwiron:
    del elwiron[elw_var]


def GetFlag(flag):
  """Runs googletest-elw-var-test_ and returns its output."""

  args = [COMMAND]
  if flag is not None:
    args += [flag]
  return gtest_test_utils.Subprocess(args, elw=elwiron).output


def TestFlag(flag, test_val, default_val):
  """Verifies that the given flag is affected by the corresponding elw var."""

  elw_var = 'GTEST_' + flag.upper()
  SetElwVar(elw_var, test_val)
  AssertEq(test_val, GetFlag(flag))
  SetElwVar(elw_var, None)
  AssertEq(default_val, GetFlag(flag))


class GTestElwVarTest(gtest_test_utils.TestCase):

  def testElwVarAffectsFlag(self):
    """Tests that environment variable should affect the corresponding flag."""

    TestFlag('break_on_failure', '1', '0')
    TestFlag('color', 'yes', 'auto')
    TestFlag('filter', 'FooTest.Bar', '*')
    SetElwVar('XML_OUTPUT_FILE', None)  # For 'output' test
    TestFlag('output', 'xml:tmp/foo.xml', '')
    TestFlag('print_time', '0', '1')
    TestFlag('repeat', '999', '1')
    TestFlag('throw_on_failure', '1', '0')
    TestFlag('death_test_style', 'threadsafe', 'fast')
    TestFlag('catch_exceptions', '0', '1')

    if IS_LINUX:
      TestFlag('death_test_use_fork', '1', '0')
      TestFlag('stack_trace_depth', '0', '100')


  def testXmlOutputFile(self):
    """Tests that $XML_OUTPUT_FILE affects the output flag."""

    SetElwVar('GTEST_OUTPUT', None)
    SetElwVar('XML_OUTPUT_FILE', 'tmp/bar.xml')
    AssertEq('xml:tmp/bar.xml', GetFlag('output'))

  def testXmlOutputFileOverride(self):
    """Tests that $XML_OUTPUT_FILE is overridden by $GTEST_OUTPUT."""

    SetElwVar('GTEST_OUTPUT', 'xml:tmp/foo.xml')
    SetElwVar('XML_OUTPUT_FILE', 'tmp/bar.xml')
    AssertEq('xml:tmp/foo.xml', GetFlag('output'))

if __name__ == '__main__':
  gtest_test_utils.Main()
