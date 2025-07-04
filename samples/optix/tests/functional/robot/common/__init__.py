#  Copyright 2008-2012 Nokia Siemens Networks Oyj
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

"""Base classes for test exelwtion model.

This code was earlier used also by test result processing modules but not
anymore in RF 2.7.

The whole package is likely to be removed in RF 2.8 when test exelwtion model
is refactored. No new code should depend on this package.
"""

from .model import BaseTestSuite, BaseTestCase
from .keyword import BaseKeyword
from .handlers import UserErrorHandler
from .libraries import BaseLibrary
from .statistics import Statistics
