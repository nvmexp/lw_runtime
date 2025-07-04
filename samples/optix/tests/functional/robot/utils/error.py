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

import re
import sys
import traceback

from robot.errors import RobotError

from .unic import unic

RERAISED_EXCEPTIONS = (KeyboardInterrupt, SystemExit, MemoryError)
if sys.platform.startswith('java'):
    from java.io import StringWriter, PrintWriter
    from java.lang import Throwable, OutOfMemoryError
    RERAISED_EXCEPTIONS += (OutOfMemoryError,)
else:
    Throwable = ()


def get_error_message():
    """Returns error message of the last oclwrred exception.

    This method handles also exceptions containing unicode messages. Thus it
    MUST be used to get messages from all exceptions originating outside the
    framework.
    """
    return ErrorDetails().message


def get_error_details():
    """Returns error message and details of the last oclwrred exception.
    """
    details = ErrorDetails()
    return details.message, details.traceback


def ErrorDetails():
    """This factory returns an object that wraps the last oclwrred exception

    It has attributes `message`, `traceback` and `error`, where `message`
    contains type and message of the original error, `traceback` contains the
    traceback/stack trace and `error` contains the original error instance.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type in RERAISED_EXCEPTIONS:
        raise exc_value
    details = PythonErrorDetails \
            if not isinstance(exc_value, Throwable) else JavaErrorDetails
    return details(exc_type, exc_value, exc_traceback)


class _ErrorDetails(object):
    _generic_exceptions = ('AssertionError', 'AssertionFailedError', 'Exception',
                           'Error', 'RuntimeError', 'RuntimeException',
                           'DataError', 'TimeoutError', 'RemoteError')

    def __init__(self, exc_type, exc_value, exc_traceback):
        self.error = exc_value
        self._exc_value = exc_value
        self._exc_type = exc_type
        self._exc_traceback = exc_traceback
        self._message = None
        self._traceback = None

    @property
    def message(self):
        if self._message is None:
            self._message = self._get_message()
        return self._message

    @property
    def traceback(self):
        if self._traceback is None:
            self._traceback = self._get_details()
        return self._traceback

    def _get_name(self, exc_type):
        try:
            return exc_type.__name__
        except AttributeError:
            return unic(exc_type)

    def _format_message(self, name, message):
        message = unic(message or '')
        message = self._clean_up_message(message, name)
        name = name.split('.')[-1]  # Use only last part of the name
        if not message:
            return name
        if name in self._generic_exceptions or \
                getattr(self.error, 'ROBOT_SUPPRESS_NAME', False):
            return message
        return '%s: %s' % (name, message)

    def _clean_up_message(self, message, name):
        return message


class PythonErrorDetails(_ErrorDetails):

    def _get_message(self):
        # If exception is a "string exception" without a message exc_value is None
        if self._exc_value is None:
            return unic(self._exc_type)
        name = self._get_name(self._exc_type)
        try:
            msg = unicode(self._exc_value)
        except UnicodeError:  # Happens if message is Unicode and version < 2.6
            msg = ' '.join(unic(a) for a in self._exc_value.args)
        return self._format_message(name, msg)

    def _get_details(self):
        if isinstance(self._exc_value, RobotError):
            return self._exc_value.details
        return 'Traceback (most recent call last):\n' + self._get_traceback()

    def _get_traceback(self):
        tb = self._exc_traceback
        while tb and self._is_excluded_traceback(tb):
            tb = tb.tb_next
        return ''.join(traceback.format_tb(tb)).rstrip() or '  None'

    def _is_excluded_traceback(self, traceback):
        module = traceback.tb_frame.f_globals.get('__name__')
        return module and module.startswith('robot.')


class JavaErrorDetails(_ErrorDetails):
    _java_trace_re = re.compile('^\s+at (\w.+)')
    _ignored_java_trace = ('org.python.', 'robot.running.', 'robot$py.',
                           'sun.reflect.', 'java.lang.reflect.')

    def _get_message(self):
        exc_name = self._get_name(self._exc_type)
        # OOME.getMessage and even toString seem to throw NullPointerException
        if not self._is_out_of_memory_error(self._exc_type):
            exc_msg = self._exc_value.getMessage()
        else:
            exc_msg = str(self._exc_value)
        return self._format_message(exc_name, exc_msg)

    def _is_out_of_memory_error(self, exc_type):
        return exc_type is OutOfMemoryError

    def _get_details(self):
        # OOME.printStackTrace seems to throw NullPointerException
        if self._is_out_of_memory_error(self._exc_type):
            return ''
        output = StringWriter()
        self._exc_value.printStackTrace(PrintWriter(output))
        details = '\n'.join(line for line in output.toString().splitlines()
                            if not self._is_ignored_stack_trace_line(line))
        msg = unic(self._exc_value.getMessage() or '')
        if msg:
            details = details.replace(msg, '', 1)
        return details

    def _is_ignored_stack_trace_line(self, line):
        if not line:
            return True
        res = self._java_trace_re.match(line)
        if res is None:
            return False
        location = res.group(1)
        for entry in self._ignored_java_trace:
            if location.startswith(entry):
                return True
        return False

    def _clean_up_message(self, msg, name):
        msg = self._remove_stack_trace_lines(msg)
        return self._remove_exception_name(msg, name).strip()

    def _remove_stack_trace_lines(self, msg):
        lines = msg.splitlines()
        while lines:
            if self._java_trace_re.match(lines[-1]):
                lines.pop()
            else:
                break
        return '\n'.join(lines)

    def _remove_exception_name(self, msg, name):
        tokens = msg.split(':', 1)
        if len(tokens) == 2 and tokens[0] == name:
            msg = tokens[1]
        return msg
