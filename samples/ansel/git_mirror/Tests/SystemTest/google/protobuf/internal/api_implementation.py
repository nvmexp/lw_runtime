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

"""Determine which implementation of the protobuf API is used in this process.
"""

import os
import warnings
import sys

try:
  # pylint: disable=g-import-not-at-top
  from google.protobuf.internal import _api_implementation
  # The compile-time constants in the _api_implementation module can be used to
  # switch to a certain implementation of the Python API at build time.
  _api_version = _api_implementation.api_version
  _proto_extension_modules_exist_in_build = True
except ImportError:
  _api_version = -1  # Unspecified by compiler flags.
  _proto_extension_modules_exist_in_build = False

if _api_version == 1:
  raise ValueError('api_version=1 is no longer supported.')
if _api_version < 0:  # Still unspecified?
  try:
    # The presence of this module in a build allows the proto implementation to
    # be upgraded merely via build deps rather than a compiler flag or the
    # runtime environment variable.
    # pylint: disable=g-import-not-at-top
    from google.protobuf import _use_fast_cpp_protos
    # Work around a known issue in the classic bootstrap .par import hook.
    if not _use_fast_cpp_protos:
      raise ImportError('_use_fast_cpp_protos import succeeded but was None')
    del _use_fast_cpp_protos
    _api_version = 2
  except ImportError:
    try:
      # pylint: disable=g-import-not-at-top
      from google.protobuf.internal import use_pure_python
      del use_pure_python  # Avoids a pylint error and namespace pollution.
    except ImportError:
      if _proto_extension_modules_exist_in_build:
        if sys.version_info[0] >= 3:  # Python 3 defaults to C++ impl v2.
          _api_version = 2
          # TODO(b/17427486): Make Python 2 default to C++ impl v2.

_default_implementation_type = (
    'python' if _api_version <= 0 else 'cpp')

# This environment variable can be used to switch to a certain implementation
# of the Python API, overriding the compile-time constants in the
# _api_implementation module. Right now only 'python' and 'cpp' are valid
# values. Any other value will be ignored.
_implementation_type = os.getelw('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION',
                                 _default_implementation_type)

if _implementation_type != 'python':
  _implementation_type = 'cpp'

if 'PyPy' in sys.version and _implementation_type == 'cpp':
  warnings.warn('PyPy does not work yet with cpp protocol buffers. '
                'Falling back to the python implementation.')
  _implementation_type = 'python'

# This environment variable can be used to switch between the two
# 'cpp' implementations, overriding the compile-time constants in the
# _api_implementation module. Right now only '2' is supported. Any other
# value will cause an error to be raised.
_implementation_version_str = os.getelw(
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION', '2')

if _implementation_version_str != '2':
  raise ValueError(
      'unsupported PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION: "' +
      _implementation_version_str + '" (supported versions: 2)'
      )

_implementation_version = int(_implementation_version_str)


# Detect if serialization should be deterministic by default
try:
  # The presence of this module in a build allows the proto implementation to
  # be upgraded merely via build deps.
  #
  # NOTE: Merely importing this automatically enables deterministic proto
  # serialization for C++ code, but we still need to export it as a boolean so
  # that we can do the same for `_implementation_type == 'python'`.
  #
  # NOTE2: It is possible for C++ code to enable deterministic serialization by
  # default _without_ affecting Python code, if the C++ implementation is not in
  # use by this module.  That is intended behavior, so we don't actually expose
  # this boolean outside of this module.
  #
  # pylint: disable=g-import-not-at-top,unused-import
  from google.protobuf import enable_deterministic_proto_serialization
  _python_deterministic_proto_serialization = True
except ImportError:
  _python_deterministic_proto_serialization = False


# Usage of this function is discouraged. Clients shouldn't care which
# implementation of the API is in use. Note that there is no guarantee
# that differences between APIs will be maintained.
# Please don't use this function if possible.
def Type():
  return _implementation_type


# See comment on 'Type' above.
def Version():
  return _implementation_version


# For internal use only
def IsPythonDefaultSerializationDeterministic():
  return _python_deterministic_proto_serialization

# DO NOT USE: For migration and testing only. Will be removed when Proto3
# defaults to preserve unknowns.
if _implementation_type == 'cpp':
  try:
    # pylint: disable=g-import-not-at-top
    from google.protobuf.pyext import _message

    def GetPythonProto3PreserveUnknownsDefault():
      return _message.GetPythonProto3PreserveUnknownsDefault()

    def SetPythonProto3PreserveUnknownsDefault(preserve):
      _message.SetPythonProto3PreserveUnknownsDefault(preserve)
  except ImportError:
    # Unrecognized cpp implementation. Skipping the unknown fields APIs.
    pass
else:
  _python_proto3_preserve_unknowns_default = True

  def GetPythonProto3PreserveUnknownsDefault():
    return _python_proto3_preserve_unknowns_default

  def SetPythonProto3PreserveUnknownsDefault(preserve):
    global _python_proto3_preserve_unknowns_default
    _python_proto3_preserve_unknowns_default = preserve

