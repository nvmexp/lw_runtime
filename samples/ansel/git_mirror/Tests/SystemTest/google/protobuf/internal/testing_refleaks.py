# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
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

"""A subclass of unittest.TestCase which checks for reference leaks.

To use:
- Use testing_refleak.BaseTestCase instead of unittest.TestCase
- Configure and compile Python with --with-pydebug

If sys.gettotalrefcount() is not available (because Python was built without
the Py_DEBUG option), then this module is a no-op and tests will run normally.
"""

import gc
import sys

try:
  import copy_reg as copyreg  #PY26
except ImportError:
  import copyreg

try:
  import unittest2 as unittest  #PY26
except ImportError:
  import unittest


class LocalTestResult(unittest.TestResult):
  """A TestResult which forwards events to a parent object, except for Skips."""

  def __init__(self, parent_result):
    unittest.TestResult.__init__(self)
    self.parent_result = parent_result

  def addError(self, test, error):
    self.parent_result.addError(test, error)

  def addFailure(self, test, error):
    self.parent_result.addFailure(test, error)

  def addSkip(self, test, reason):
    pass


class ReferenceLeakCheckerTestCase(unittest.TestCase):
  """A TestCase which runs tests multiple times, collecting reference counts."""

  NB_RUNS = 3

  def run(self, result=None):
    # python_message.py registers all Message classes to some pickle global
    # registry, which makes the classes immortal.
    # We save a copy of this registry, and reset it before we could references.
    self._saved_pickle_registry = copyreg.dispatch_table.copy()

    # Run the test twice, to warm up the instance attributes.
    super(ReferenceLeakCheckerTestCase, self).run(result=result)
    super(ReferenceLeakCheckerTestCase, self).run(result=result)

    oldrefcount = 0
    local_result = LocalTestResult(result)

    refcount_deltas = []
    for _ in range(self.NB_RUNS):
      oldrefcount = self._getRefcounts()
      super(ReferenceLeakCheckerTestCase, self).run(result=local_result)
      newrefcount = self._getRefcounts()
      refcount_deltas.append(newrefcount - oldrefcount)
    print(refcount_deltas, self)

    try:
      self.assertEqual(refcount_deltas, [0] * self.NB_RUNS)
    except Exception:  # pylint: disable=broad-except
      result.addError(self, sys.exc_info())

  def _getRefcounts(self):
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(self._saved_pickle_registry)
    # It is sometimes necessary to gc.collect() multiple times, to ensure
    # that all objects can be collected.
    gc.collect()
    gc.collect()
    gc.collect()
    return sys.gettotalrefcount()


if hasattr(sys, 'gettotalrefcount'):
  BaseTestCase = ReferenceLeakCheckerTestCase
  SkipReferenceLeakChecker = unittest.skip

else:
  # When PyDEBUG is not enabled, run the tests normally.
  BaseTestCase = unittest.TestCase

  def SkipReferenceLeakChecker(reason):
    del reason  # Don't skip, so don't need a reason.
    def Same(func):
      return func
    return Same
