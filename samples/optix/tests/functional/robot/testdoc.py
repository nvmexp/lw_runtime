#!/usr/bin/elw python

#  Copyright 2008-2014 Nokia Solutions and Networks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Module implementing the command line entry point for the `Testdoc` tool.

This module can be exelwted from the command line using the following
approaches::

    python -m robot.testdoc
    python path/to/robot/testdoc.py

Instead of ``python`` it is possible to use also other Python interpreters.

This module also provides :func:`testdoc` and :func:`testdoc_cli` functions
that can be used programmatically. Other code is for internal usage.
"""

from __future__ import with_statement

USAGE = """robot.testdoc -- Robot Framework test data documentation tool

Version:  <VERSION>

Usage:  python -m robot.testdoc [options] data_sources output_file

Testdoc generates a high level test documentation based on Robot Framework
test data. Generated documentation includes name, documentation and other
metadata of each test suite and test case, as well as the top-level keywords
and their arguments.

Options
=======

  -T --title title       Set the title of the generated documentation.
                         Underscores in the title are colwerted to spaces.
                         The default title is the name of the top level suite.
  -N --name name         Override the name of the top level suite.
  -D --doc document      Override the documentation of the top level suite.
  -M --metadata name:value *  Set/override metadata of the top level suite.
  -G --settag tag *      Set given tag(s) to all test cases.
  -t --test name *       Include tests by name.
  -s --suite name *      Include suites by name.
  -i --include tag *     Include tests by tags.
  -e --exclude tag *     Exclude tests by tags.
  -h -? --help           Print this help.

All options except --title have exactly same semantics as same options have
when exelwting test cases.

Exelwtion
=========

Data can be given as a single file, directory, or as multiple files and
directories. In all these cases, the last argument must be the file where
to write the output. The output is always created in HTML format.

Testdoc works with all interpreters supported by Robot Framework (Python,
Jython and IronPython). It can be exelwted as an installed module like
`python -m robot.testdoc` or as a script like `python path/robot/testdoc.py`.

Examples:

  python -m robot.testdoc my_test.html testdoc.html
  jython -m robot.testdoc -N smoke_tests -i smoke path/to/my_tests smoke.html
  ipy path/to/robot/testdoc.py first_suite.txt second_suite.txt output.html

For more information about Testdoc and other built-in tools, see
http://robotframework.org/robotframework/#built-in-tools.
"""

import os.path
import sys
import time

# Allows running as a script. __name__ check needed with multiprocessing:
# http://code.google.com/p/robotframework/issues/detail?id=1137
if 'robot' not in sys.modules and __name__ == '__main__':
    import pythonpathsetter

from robot import utils
from robot.conf import RobotSettings
from robot.htmldata import HtmlFileWriter, ModelWriter, JsonWriter, TESTDOC
from robot.parsing import disable_lwrdir_processing
from robot.running import TestSuiteBuilder


class TestDoc(utils.Application):

    def __init__(self):
        utils.Application.__init__(self, USAGE, arg_limits=(2,))

    def main(self, datasources, title=None, **options):
        outfile = utils.abspath(datasources.pop())
        suite = TestSuiteFactory(datasources, **options)
        self._write_test_doc(suite, outfile, title)
        self.console(outfile)

    def _write_test_doc(self, suite, outfile, title):
        with open(outfile, 'w') as output:
            model_writer = TestdocModelWriter(output, suite, title)
            HtmlFileWriter(output, model_writer).write(TESTDOC)


@disable_lwrdir_processing
def TestSuiteFactory(datasources, **options):
    settings = RobotSettings(options)
    if isinstance(datasources, basestring):
        datasources = [datasources]
    suite = TestSuiteBuilder().build(*datasources)
    suite.configure(**settings.suite_config)
    return suite


class TestdocModelWriter(ModelWriter):

    def __init__(self, output, suite, title=None):
        self._output = output
        self._output_path = getattr(output, 'name', None)
        self._suite = suite
        self._title = title.replace('_', ' ') if title else suite.name

    def write(self, line):
        self._output.write('<script type="text/javascript">\n')
        self.write_data()
        self._output.write('</script>\n')

    def write_data(self):
        generated_time = time.localtime()
        model = {
            'suite': JsonColwerter(self._output_path).colwert(self._suite),
            'title': self._title,
            'generated': utils.format_time(generated_time, gmtsep=' '),
            'generatedMillis': long(time.mktime(generated_time) * 1000)
        }
        JsonWriter(self._output).write_json('testdoc = ', model)


class JsonColwerter(object):

    def __init__(self, output_path=None):
        self._output_path = output_path

    def colwert(self, suite):
        return self._colwert_suite(suite)

    def _colwert_suite(self, suite):
        return {
            'source': suite.source or '',
            'relativeSource': self._get_relative_source(suite.source),
            'id': suite.id,
            'name': self._escape(suite.name),
            'fullName': self._escape(suite.longname),
            'doc': self._html(suite.doc),
            'metadata': [(self._escape(name), self._html(value))
                         for name, value in suite.metadata.items()],
            'numberOfTests': suite.test_count   ,
            'suites': self._colwert_suites(suite),
            'tests': self._colwert_tests(suite),
            'keywords': list(self._colwert_keywords(suite))
        }

    def _get_relative_source(self, source):
        if not source or not self._output_path:
            return ''
        return utils.get_link_path(source, os.path.dirname(self._output_path))

    def _escape(self, item):
        return utils.html_escape(item)

    def _html(self, item):
        return utils.html_format(utils.unescape(item))

    def _colwert_suites(self, suite):
        return [self._colwert_suite(s) for s in suite.suites]

    def _colwert_tests(self, suite):
        return [self._colwert_test(t) for t in suite.tests]

    def _colwert_test(self, test):
        return {
            'name': self._escape(test.name),
            'fullName': self._escape(test.longname),
            'id': test.id,
            'doc': self._html(test.doc),
            'tags': [self._escape(t) for t in test.tags],
            'timeout': self._get_timeout(test.timeout),
            'keywords': list(self._colwert_keywords(test))
        }

    def _colwert_keywords(self, item):
        for kw in getattr(item, 'keywords', []):
            if kw.type == 'setup':
                yield self._colwert_keyword(kw, 'SETUP')
            elif kw.type == 'teardown':
                yield self._colwert_keyword(kw, 'TEARDOWN')
            elif kw.is_for_loop():
                yield self._colwert_for_loop(kw)
            else:
                yield self._colwert_keyword(kw, 'KEYWORD')

    def _colwert_for_loop(self, kw):
        return {
            'name': self._escape(self._get_for_loop(kw)),
            'arguments': '',
            'type': 'FOR'
        }

    def _colwert_keyword(self, kw, kw_type):
        return {
            'name': self._escape(self._get_kw_name(kw)),
            'arguments': self._escape(', '.join(kw.args)),
            'type': kw_type
        }

    def _get_kw_name(self, kw):
        if kw.assign:
            return '%s = %s' % (', '.join(a.rstrip('= ') for a in kw.assign), kw.name)
        return kw.name

    def _get_for_loop(self, kw):
        joiner = ' IN RANGE ' if kw.range else ' IN '
        return ', '.join(kw.vars) + joiner + utils.seq2str2(kw.items)

    def _get_timeout(self, timeout):
        if timeout is None:
            return ''
        try:
            tout = utils.secs_to_timestr(utils.timestr_to_secs(timeout.value))
        except ValueError:
            tout = timeout.value
        if timeout.message:
            tout += ' :: ' + timeout.message
        return tout


def testdoc_cli(arguments):
    """Exelwtes `Testdoc` similarly as from the command line.

    :param arguments: command line arguments as a list of strings.

    For programmatic usage the :func:`testdoc` function is typically better. It
    has a better API for that and does not call :func:`sys.exit` like
    this function.

    Example::

        from robot.testdoc import testdoc_cli

        testdoc_cli(['--title', 'Test Plan', 'mytests', 'plan.html'])
    """
    TestDoc().exelwte_cli(arguments)


def testdoc(*arguments, **options):
    """Exelwtes `Testdoc` programmatically.

    Arguments and options have same semantics, and options have same names,
    as arguments and options to Testdoc.

    Example::

        from robot.testdoc import testdoc

        testdoc('mytests', 'plan.html', title='Test Plan')
    """
    TestDoc().execute(*arguments, **options)


if __name__ == '__main__':
    testdoc_cli(sys.argv[1:])
