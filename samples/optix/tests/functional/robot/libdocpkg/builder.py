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
import os

from robot.parsing import VALID_EXTENSIONS as RESOURCE_EXTENSIONS
from robot.errors import DataError

from .robotbuilder import LibraryDocBuilder, ResourceDocBuilder
from .specbuilder import SpecDocBuilder
if sys.platform.startswith('java'):
    from .javabuilder import JavaDocBuilder
else:
    def JavaDocBuilder():
        raise DataError('Dolwmenting Java test libraries requires Jython.')


def DolwmentationBuilder(library_or_resource):
    extension = os.path.splitext(library_or_resource)[1][1:].lower()
    if extension in RESOURCE_EXTENSIONS:
        return ResourceDocBuilder()
    if extension == 'xml':
        return SpecDocBuilder()
    if extension == 'java':
        return JavaDocBuilder()
    return LibraryDocBuilder()
