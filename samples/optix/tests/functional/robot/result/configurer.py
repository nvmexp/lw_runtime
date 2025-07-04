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

from robot import utils
from robot import model


class SuiteConfigurer(model.SuiteConfigurer):
    """Result suite configured.

    Calls suite's
    :meth:`~robot.result.testsuite.TestSuite.remove_keywords`,
    :meth:`~robot.result.testsuite.TestSuite.filter_messages` and
    :meth:`~robot.result.testsuite.TestSuite.set_criticality` methods
    and sets it's start and end time based on the given named parameters.

    ``base_config`` is forwarded to
    :class:`robot.model.SuiteConfigurer <robot.model.configurer.SuiteConfigurer>`
    that will do further configuration based on them.
    """

    def __init__(self, remove_keywords=None, log_level=None, start_time=None,
                 end_time=None, critical_tags=None, non_critical_tags=None,
                 **base_config):
        model.SuiteConfigurer.__init__(self, **base_config)
        self.remove_keywords = self._get_remove_keywords(remove_keywords)
        self.log_level = log_level
        self.start_time = self._get_time(start_time)
        self.end_time = self._get_time(end_time)
        self.critical_tags = critical_tags
        self.non_critical_tags = non_critical_tags

    def _get_remove_keywords(self, value):
        if value is None:
            return []
        if isinstance(value, basestring):
            return [value]
        return value

    def _get_time(self, timestamp):
        if not timestamp:
            return None
        try:
            secs = utils.timestamp_to_secs(timestamp, seps=' :.-_')
        except ValueError:
            return None
        return utils.secs_to_timestamp(secs, millis=True)

    def visit_suite(self, suite):
        model.SuiteConfigurer.visit_suite(self, suite)
        self._remove_keywords(suite)
        self._set_times(suite)
        suite.filter_messages(self.log_level)
        suite.set_criticality(self.critical_tags, self.non_critical_tags)

    def _remove_keywords(self, suite):
        for how in self.remove_keywords:
            suite.remove_keywords(how)

    def _set_times(self, suite):
        if self.start_time:
            suite.starttime = self.start_time
        if self.end_time:
            suite.endtime = self.end_time
