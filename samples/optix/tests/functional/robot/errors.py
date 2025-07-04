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

import utils

# Return codes from Robot and Rebot.
# RC below 250 is the number of failed critical tests and exactly 250
# means that number or more such failures.
INFO_PRINTED    = 251   # --help or --version
DATA_ERROR      = 252   # Invalid data or cli args
STOPPED_BY_USER = 253   # KeyboardInterrupt or SystemExit
FRAMEWORK_ERROR = 255   # Unexpected error


class RobotError(Exception):
    """Base class for Robot Framework errors.

    Do not raise this method but use more specific errors instead.
    """
    def __init__(self, message='', details=''):
        Exception.__init__(self, message)
        self.details = details

    @property
    def message(self):
        return self.__unicode__()

    def __unicode__(self):
        # Needed to handle exceptions w/ Unicode correctly on Python 2.5
        return unicode(self.args[0]) if self.args else u''


class FrameworkError(RobotError):
    """Can be used when the core framework goes to unexpected state.

    It is good to explicitly raise a FrameworkError if some framework
    component is used incorrectly. This is pretty much same as
    'Internal Error' and should of course never happen.
    """


class DataError(RobotError):
    """Used when the provided test data is invalid.

    DataErrors are not be caught by keywords that run other keywords
    (e.g. `Run Keyword And Expect Error`). Libraries should thus use
    this exception with care.
    """


class TimeoutError(RobotError):
    """Used when a test or keyword timeout oclwrs.

    This exception is handled specially so that exelwtion of the
    current test is always stopped immediately and it is not caught by
    keywords exelwting other keywords (e.g. `Run Keyword And Expect
    Error`). Libraries should thus NOT use this exception themselves.
    """


class Information(RobotError):
    """Used by argument parser with --help or --version."""


class ExelwtionFailed(RobotError):
    """Used for communicating failures in test exelwtion."""

    def __init__(self, message, timeout=False, syntax=False, exit=False,
                 continue_on_failure=False, return_value=None):
        if '\r\n' in message:
            message = message.replace('\r\n', '\n')
        RobotError.__init__(self, utils.lwt_long_message(message))
        self.timeout = timeout
        self.syntax = syntax
        self.exit = exit
        self.continue_on_failure = continue_on_failure
        self.return_value = return_value

    @property
    def dont_continue(self):
        return self.timeout or self.syntax or self.exit

    def _get_continue_on_failure(self):
        return self._continue_on_failure

    def _set_continue_on_failure(self, continue_on_failure):
        self._continue_on_failure = continue_on_failure
        for child in getattr(self, '_errors', []):
            child.continue_on_failure = continue_on_failure

    continue_on_failure = property(_get_continue_on_failure,
                                   _set_continue_on_failure)

    def can_continue(self, teardown=False, templated=False, dry_run=False):
        if dry_run:
            return True
        if self.dont_continue and not (teardown and self.syntax):
            return False
        if teardown or templated:
            return True
        return self.continue_on_failure

    def get_errors(self):
        return [self]


class HandlerExelwtionFailed(ExelwtionFailed):

    def __init__(self):
        details = utils.ErrorDetails()
        timeout = isinstance(details.error, TimeoutError)
        syntax = isinstance(details.error, DataError)
        exit_on_failure = self._get(details.error, 'EXIT_ON_FAILURE')
        continue_on_failure = self._get(details.error, 'CONTINUE_ON_FAILURE')
        ExelwtionFailed.__init__(self, details.message, timeout, syntax,
                                 exit_on_failure, continue_on_failure)
        self.full_message = details.message
        self.traceback = details.traceback
        self._handle_deprecated_exit_for_loop(details.error)

    def _get(self, error, attr):
        return bool(getattr(error, 'ROBOT_' + attr, False))

    def _handle_deprecated_exit_for_loop(self, error):
        if self._get(error, 'EXIT_FOR_LOOP'):
            from robot.output import LOGGER
            LOGGER.warn("Support for using 'ROBOT_EXIT_FOR_LOOP' attribute to "
                        "exit for loops is deprecated in Robot Framework 2.8 "
                        "and will be removed in 2.9.")
            raise ExitForLoop


class ExelwtionFailures(ExelwtionFailed):

    def __init__(self, errors, message=None):
        message = message or self._format_message([unicode(e) for e in errors])
        ExelwtionFailed.__init__(self, message, **self._get_attrs(errors))
        self._errors = errors

    def _format_message(self, messages):
        if len(messages) == 1:
            return messages[0]
        lines = ['Several failures oclwrred:'] \
                + ['%d) %s' % (i+1, m) for i, m in enumerate(messages)]
        return '\n\n'.join(lines)

    def _get_attrs(self, errors):
        return {'timeout': any(err.timeout for err in errors),
                'syntax': any(err.syntax for err in errors),
                'exit': any(err.exit for err in errors),
                'continue_on_failure': all(err.continue_on_failure for err in errors)
                }

    def get_errors(self):
        return self._errors


class UserKeywordExelwtionFailed(ExelwtionFailures):

    def __init__(self, run_errors=None, teardown_errors=None):
        errors = self._get_active_errors(run_errors, teardown_errors)
        message = self._get_message(run_errors, teardown_errors)
        ExelwtionFailures.__init__(self, errors, message)
        if run_errors and not teardown_errors:
            self._errors = run_errors.get_errors()
        else:
            self._errors = [self]

    def _get_active_errors(self, *errors):
        return [err for err in errors if err]

    def _get_message(self, run_errors, teardown_errors):
        run_msg = unicode(run_errors or '')
        td_msg = unicode(teardown_errors or '')
        if not td_msg:
            return run_msg
        if not run_msg:
            return 'Keyword teardown failed:\n%s' % td_msg
        return '%s\n\nAlso keyword teardown failed:\n%s' % (run_msg, td_msg)


class ExelwtionPassed(ExelwtionFailed):
    """Base class for all exceptions communicating that exelwtion passed.

    Should not be raised directly, but more detailed exceptions used instead.
    """

    def __init__(self, message=None, **kwargs):
        ExelwtionFailed.__init__(self, message or self._get_message(), **kwargs)
        self._earlier_failures = []

    def _get_message(self):
        return "Invalid '%s' usage." \
               % utils.printable_name(self.__class__.__name__, code_style=True)

    def set_earlier_failures(self, failures):
        if failures:
            self._earlier_failures.extend(failures)

    @property
    def earlier_failures(self):
        if not self._earlier_failures:
            return None
        return ExelwtionFailures(self._earlier_failures)


class PassExelwtion(ExelwtionPassed):
    """Used by 'Pass Exelwtion' keyword."""

    def __init__(self, message):
        ExelwtionPassed.__init__(self, message)


class ContinueForLoop(ExelwtionPassed):
    """Used by 'Continue For Loop' keyword."""


class ExitForLoop(ExelwtionPassed):
    """Used by 'Exit For Loop' keyword."""


class ReturnFromKeyword(ExelwtionPassed):
    """Used by 'Return From Keyword' keyword."""

    def __init__(self, return_value):
        ExelwtionPassed.__init__(self, return_value=return_value)


class RemoteError(RobotError):
    """Used by Remote library to report remote errors."""

    def __init__(self, message='', details='', fatal=False, continuable=False):
        RobotError.__init__(self, message, details)
        self.ROBOT_EXIT_ON_FAILURE = fatal
        self.ROBOT_CONTINUE_ON_FAILURE = continuable
