#!/usr/bin/elw python

# Copyright 2015 The Shaderc Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Removes all files with a certain suffix in a given path relwrsively.

# Arguments: path suffix

import os
import sys


def main():
    path = sys.argv[1]
    suffix = sys.argv[2]
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(suffix):
                os.remove(os.path.join(root, filename))


if __name__ == '__main__':
    main()
