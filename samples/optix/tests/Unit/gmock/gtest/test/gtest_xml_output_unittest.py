#!/usr/bin/elw python
#
# Copyright 2006, Google Inc.
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

"""Unit test for the gtest_xml_output module"""

__author__ = 'eefacm@gmail.com (Sean Mcafee)'

import datetime
import errno
import os
import re
import sys
from xml.dom import minidom, Node

import gtest_test_utils
import gtest_xml_test_utils


GTEST_FILTER_FLAG = '--gtest_filter'
GTEST_LIST_TESTS_FLAG = '--gtest_list_tests'
GTEST_OUTPUT_FLAG         = "--gtest_output"
GTEST_DEFAULT_OUTPUT_FILE = "test_detail.xml"
GTEST_PROGRAM_NAME = "gtest_xml_output_unittest_"

SUPPORTS_STACK_TRACES = False

if SUPPORTS_STACK_TRACES:
  STACK_TRACE_TEMPLATE = '\nStack trace:\n*'
else:
  STACK_TRACE_TEMPLATE = ''

EXPECTED_NON_EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="23" failures="4" disabled="2" errors="0" time="*" timestamp="*" name="AllTests" ad_hoc_property="42">
  <testsuite name="SuccessfulTest" tests="1" failures="0" disabled="0" errors="0" time="*">
    <testcase name="Succeeds" status="run" time="*" classname="SuccessfulTest"/>
  </testsuite>
  <testsuite name="FailedTest" tests="1" failures="1" disabled="0" errors="0" time="*">
    <testcase name="Fails" status="run" time="*" classname="FailedTest">
      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Value of: 2&#x0A;Expected: 1" type=""><![CDATA[gtest_xml_output_unittest_.cc:*
Value of: 2
Expected: 1%(stack)s]]></failure>
    </testcase>
  </testsuite>
  <testsuite name="MixedResultTest" tests="3" failures="1" disabled="1" errors="0" time="*">
    <testcase name="Succeeds" status="run" time="*" classname="MixedResultTest"/>
    <testcase name="Fails" status="run" time="*" classname="MixedResultTest">
      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Value of: 2&#x0A;Expected: 1" type=""><![CDATA[gtest_xml_output_unittest_.cc:*
Value of: 2
Expected: 1%(stack)s]]></failure>
      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Value of: 3&#x0A;Expected: 2" type=""><![CDATA[gtest_xml_output_unittest_.cc:*
Value of: 3
Expected: 2%(stack)s]]></failure>
    </testcase>
    <testcase name="DISABLED_test" status="notrun" time="*" classname="MixedResultTest"/>
  </testsuite>
  <testsuite name="XmlQuotingTest" tests="1" failures="1" disabled="0" errors="0" time="*">
    <testcase name="OutputsCData" status="run" time="*" classname="XmlQuotingTest">
      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Failed&#x0A;XML output: &lt;?xml encoding=&quot;utf-8&quot;&gt;&lt;top&gt;&lt;![CDATA[cdata text]]&gt;&lt;/top&gt;" type=""><![CDATA[gtest_xml_output_unittest_.cc:*
Failed
XML output: <?xml encoding="utf-8"><top><![CDATA[cdata text]]>]]&gt;<![CDATA[</top>%(stack)s]]></failure>
    </testcase>
  </testsuite>
  <testsuite name="IlwalidCharactersTest" tests="1" failures="1" disabled="0" errors="0" time="*">
    <testcase name="IlwalidCharactersInMessage" status="run" time="*" classname="IlwalidCharactersTest">
      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Failed&#x0A;Invalid characters in brackets []" type=""><![CDATA[gtest_xml_output_unittest_.cc:*
Failed
Invalid characters in brackets []%(stack)s]]></failure>
    </testcase>
  </testsuite>
  <testsuite name="DisabledTest" tests="1" failures="0" disabled="1" errors="0" time="*">
    <testcase name="DISABLED_test_not_run" status="notrun" time="*" classname="DisabledTest"/>
  </testsuite>
  <testsuite name="PropertyRecordingTest" tests="4" failures="0" disabled="0" errors="0" time="*" SetUpTestCase="yes" TearDownTestCase="aye">
    <testcase name="OneProperty" status="run" time="*" classname="PropertyRecordingTest" key_1="1"/>
    <testcase name="IntValuedProperty" status="run" time="*" classname="PropertyRecordingTest" key_int="1"/>
    <testcase name="ThreeProperties" status="run" time="*" classname="PropertyRecordingTest" key_1="1" key_2="2" key_3="3"/>
    <testcase name="TwoValuesForOneKeyUsesLastValue" status="run" time="*" classname="PropertyRecordingTest" key_1="2"/>
  </testsuite>
  <testsuite name="NoFixtureTest" tests="3" failures="0" disabled="0" errors="0" time="*">
     <testcase name="RecordProperty" status="run" time="*" classname="NoFixtureTest" key="1"/>
     <testcase name="ExternalUtilityThatCallsRecordIntValuedProperty" status="run" time="*" classname="NoFixtureTest" key_for_utility_int="1"/>
     <testcase name="ExternalUtilityThatCallsRecordStringValuedProperty" status="run" time="*" classname="NoFixtureTest" key_for_utility_string="1"/>
  </testsuite>
  <testsuite name="Single/ValueParamTest" tests="4" failures="0" disabled="0" errors="0" time="*">
    <testcase name="HasValueParamAttribute/0" value_param="33" status="run" time="*" classname="Single/ValueParamTest" />
    <testcase name="HasValueParamAttribute/1" value_param="42" status="run" time="*" classname="Single/ValueParamTest" />
    <testcase name="AnotherTestThatHasValueParamAttribute/0" value_param="33" status="run" time="*" classname="Single/ValueParamTest" />
    <testcase name="AnotherTestThatHasValueParamAttribute/1" value_param="42" status="run" time="*" classname="Single/ValueParamTest" />
  </testsuite>
  <testsuite name="TypedTest/0" tests="1" failures="0" disabled="0" errors="0" time="*">
    <testcase name="HasTypeParamAttribute" type_param="*" status="run" time="*" classname="TypedTest/0" />
  </testsuite>
  <testsuite name="TypedTest/1" tests="1" failures="0" disabled="0" errors="0" time="*">
    <testcase name="HasTypeParamAttribute" type_param="*" status="run" time="*" classname="TypedTest/1" />
  </testsuite>
  <testsuite name="Single/TypeParameterizedTestCase/0" tests="1" failures="0" disabled="0" errors="0" time="*">
    <testcase name="HasTypeParamAttribute" type_param="*" status="run" time="*" classname="Single/TypeParameterizedTestCase/0" />
  </testsuite>
  <testsuite name="Single/TypeParameterizedTestCase/1" tests="1" failures="0" disabled="0" errors="0" time="*">
    <testcase name="HasTypeParamAttribute" type_param="*" status="run" time="*" classname="Single/TypeParameterizedTestCase/1" />
  </testsuite>
</testsuites>""" % {'stack': STACK_TRACE_TEMPLATE}

EXPECTED_FILTERED_TEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="0" disabled="0" errors="0" time="*"
            timestamp="*" name="AllTests" ad_hoc_property="42">
  <testsuite name="SuccessfulTest" tests="1" failures="0" disabled="0"
             errors="0" time="*">
    <testcase name="Succeeds" status="run" time="*" classname="SuccessfulTest"/>
  </testsuite>
</testsuites>"""

EXPECTED_EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="0" failures="0" disabled="0" errors="0" time="*"
            timestamp="*" name="AllTests">
</testsuites>"""

GTEST_PROGRAM_PATH = gtest_test_utils.GetTestExelwtablePath(GTEST_PROGRAM_NAME)

SUPPORTS_TYPED_TESTS = 'TypedTest' in gtest_test_utils.Subprocess(
    [GTEST_PROGRAM_PATH, GTEST_LIST_TESTS_FLAG], capture_stderr=False).output


class GTestXMLOutputUnitTest(gtest_xml_test_utils.GTestXMLTestCase):
  """
  Unit test for Google Test's XML output functionality.
  """

  # This test lwrrently breaks on platforms that do not support typed and
  # type-parameterized tests, so we don't run it under them.
  if SUPPORTS_TYPED_TESTS:
    def testNonEmptyXmlOutput(self):
      """
      Runs a test program that generates a non-empty XML output, and
      tests that the XML output is expected.
      """
      self._TestXmlOutput(GTEST_PROGRAM_NAME, EXPECTED_NON_EMPTY_XML, 1)

  def testEmptyXmlOutput(self):
    """Verifies XML output for a Google Test binary without actual tests.

    Runs a test program that generates an empty XML output, and
    tests that the XML output is expected.
    """

    self._TestXmlOutput('gtest_no_test_unittest', EXPECTED_EMPTY_XML, 0)

  def testTimestampValue(self):
    """Checks whether the timestamp attribute in the XML output is valid.

    Runs a test program that generates an empty XML output, and checks if
    the timestamp attribute in the testsuites tag is valid.
    """
    actual = self._GetXmlOutput('gtest_no_test_unittest', [], 0)
    date_time_str = actual.dolwmentElement.getAttributeNode('timestamp').value
    # datetime.strptime() is only available in Python 2.5+ so we have to
    # parse the expected datetime manually.
    match = re.match(r'(\d+)-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)', date_time_str)
    self.assertTrue(
        re.match,
        'XML datettime string %s has incorrect format' % date_time_str)
    date_time_from_xml = datetime.datetime(
        year=int(match.group(1)), month=int(match.group(2)),
        day=int(match.group(3)), hour=int(match.group(4)),
        minute=int(match.group(5)), second=int(match.group(6)))

    time_delta = abs(datetime.datetime.now() - date_time_from_xml)
    # timestamp value should be near the current local time
    self.assertTrue(time_delta < datetime.timedelta(seconds=600),
                    'time_delta is %s' % time_delta)
    actual.unlink()

  def testDefaultOutputFile(self):
    """
    Confirms that Google Test produces an XML output file with the expected
    default name if no name is explicitly specified.
    """
    output_file = os.path.join(gtest_test_utils.GetTempDir(),
                               GTEST_DEFAULT_OUTPUT_FILE)
    gtest_prog_path = gtest_test_utils.GetTestExelwtablePath(
        'gtest_no_test_unittest')
    try:
      os.remove(output_file)
    except OSError, e:
      if e.errno != errno.ENOENT:
        raise

    p = gtest_test_utils.Subprocess(
        [gtest_prog_path, '%s=xml' % GTEST_OUTPUT_FLAG],
        working_dir=gtest_test_utils.GetTempDir())
    self.assert_(p.exited)
    self.assertEquals(0, p.exit_code)
    self.assert_(os.path.isfile(output_file))

  def testSuppressedXmlOutput(self):
    """
    Tests that no XML file is generated if the default XML listener is
    shut down before RUN_ALL_TESTS is ilwoked.
    """

    xml_path = os.path.join(gtest_test_utils.GetTempDir(),
                            GTEST_PROGRAM_NAME + 'out.xml')
    if os.path.isfile(xml_path):
      os.remove(xml_path)

    command = [GTEST_PROGRAM_PATH,
               '%s=xml:%s' % (GTEST_OUTPUT_FLAG, xml_path),
               '--shut_down_xml']
    p = gtest_test_utils.Subprocess(command)
    if p.terminated_by_signal:
      # p.signal is avalable only if p.terminated_by_signal is True.
      self.assertFalse(
          p.terminated_by_signal,
          '%s was killed by signal %d' % (GTEST_PROGRAM_NAME, p.signal))
    else:
      self.assert_(p.exited)
      self.assertEquals(1, p.exit_code,
                        "'%s' exited with code %s, which doesn't match "
                        'the expected exit code %s.'
                        % (command, p.exit_code, 1))

    self.assert_(not os.path.isfile(xml_path))

  def testFilteredTestXmlOutput(self):
    """Verifies XML output when a filter is applied.

    Runs a test program that exelwtes only some tests and verifies that
    non-selected tests do not show up in the XML output.
    """

    self._TestXmlOutput(GTEST_PROGRAM_NAME, EXPECTED_FILTERED_TEST_XML, 0,
                        extra_args=['%s=SuccessfulTest.*' % GTEST_FILTER_FLAG])

  def _GetXmlOutput(self, gtest_prog_name, extra_args, expected_exit_code):
    """
    Returns the xml output generated by running the program gtest_prog_name.
    Furthermore, the program's exit code must be expected_exit_code.
    """
    xml_path = os.path.join(gtest_test_utils.GetTempDir(),
                            gtest_prog_name + 'out.xml')
    gtest_prog_path = gtest_test_utils.GetTestExelwtablePath(gtest_prog_name)

    command = ([gtest_prog_path, '%s=xml:%s' % (GTEST_OUTPUT_FLAG, xml_path)] +
               extra_args)
    p = gtest_test_utils.Subprocess(command)
    if p.terminated_by_signal:
      self.assert_(False,
                   '%s was killed by signal %d' % (gtest_prog_name, p.signal))
    else:
      self.assert_(p.exited)
      self.assertEquals(expected_exit_code, p.exit_code,
                        "'%s' exited with code %s, which doesn't match "
                        'the expected exit code %s.'
                        % (command, p.exit_code, expected_exit_code))
    actual = minidom.parse(xml_path)
    return actual

  def _TestXmlOutput(self, gtest_prog_name, expected_xml,
                     expected_exit_code, extra_args=None):
    """
    Asserts that the XML document generated by running the program
    gtest_prog_name matches expected_xml, a string containing another
    XML document.  Furthermore, the program's exit code must be
    expected_exit_code.
    """

    actual = self._GetXmlOutput(gtest_prog_name, extra_args or [],
                                expected_exit_code)
    expected = minidom.parseString(expected_xml)
    self.NormalizeXml(actual.dolwmentElement)
    self.AssertEquivalentNodes(expected.dolwmentElement,
                               actual.dolwmentElement)
    expected.unlink()
    actual.unlink()


if __name__ == '__main__':
  os.elwiron['GTEST_STACK_TRACE_DEPTH'] = '1'
  gtest_test_utils.Main()
