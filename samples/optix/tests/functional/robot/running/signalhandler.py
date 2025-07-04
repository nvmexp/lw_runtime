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

import sys
from threading import lwrrentThread
try:
    import signal
except ImportError:
    signal = None  # IronPython 2.6 doesn't have signal module by default
if sys.platform.startswith('java'):
    from java.lang import IllegalArgumentException
else:
    IllegalArgumentException = None

from robot.errors import ExelwtionFailed
from robot.output import LOGGER


class _StopSignalMonitor(object):

    def __init__(self):
        self._signal_count = 0
        self._running_keyword = False
        self._orig_sigint = None
        self._orig_sigterm = None

    def __call__(self, signum, frame):
        self._signal_count += 1
        LOGGER.info('Received signal: %s.' % signum)
        if self._signal_count > 1:
            sys.__stderr__.write('Exelwtion forcefully stopped.\n')
            raise SystemExit()
        sys.__stderr__.write('Second signal will force exit.\n')
        if self._running_keyword and not sys.platform.startswith('java'):
            self._stop_exelwtion_gracefully()

    def _stop_exelwtion_gracefully(self):
        raise ExelwtionFailed('Exelwtion terminated by signal', exit=True)

    def start(self):
        # TODO: Remove start() in favor of __enter__ in RF 2.9. Refactoring
        # the whole signal handler at that point would be a good idea.
        self.__enter__()

    def __enter__(self):
        if self._can_register_signal:
            self._orig_sigint = signal.getsignal(signal.SIGINT)
            self._orig_sigterm = signal.getsignal(signal.SIGTERM)
            for signum in signal.SIGINT, signal.SIGTERM:
                self._register_signal_handler(signum)
        return self

    def __exit__(self, *exc_info):
        if self._can_register_signal:
            signal.signal(signal.SIGINT, self._orig_sigint)
            signal.signal(signal.SIGTERM, self._orig_sigterm)

    @property
    def _can_register_signal(self):
        return signal and lwrrentThread().getName() == 'MainThread'

    def _register_signal_handler(self, signum):
        try:
            signal.signal(signum, self)
        except (ValueError, IllegalArgumentException), err:
            # IllegalArgumentException due to http://bugs.jython.org/issue1729
            self._warn_about_registeration_error(signum, err)

    def _warn_about_registeration_error(self, signum, err):
        name, ctrlc = {signal.SIGINT: ('INT', 'or with Ctrl-C '),
                       signal.SIGTERM: ('TERM', '')}[signum]
        LOGGER.warn('Registering signal %s failed. Stopping exelwtion '
                    'gracefully with this signal %sis not possible. '
                    'Original error was: %s' % (name, ctrlc, err))

    def start_running_keyword(self, in_teardown):
        self._running_keyword = True
        if self._signal_count and not in_teardown:
            self._stop_exelwtion_gracefully()

    def stop_running_keyword(self):
        self._running_keyword = False


STOP_SIGNAL_MONITOR = _StopSignalMonitor()
