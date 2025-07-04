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

"""Descriptors essentially contain exactly the information found in a .proto
file, in types that make this information accessible in Python.
"""

__author__ = 'robinson@google.com (Will Robinson)'

import six

from google.protobuf.internal import api_implementation

_USE_C_DESCRIPTORS = False
if api_implementation.Type() == 'cpp':
  # Used by MakeDescriptor in cpp mode
  import os
  import uuid
  from google.protobuf.pyext import _message
  _USE_C_DESCRIPTORS = getattr(_message, '_USE_C_DESCRIPTORS', False)


class Error(Exception):
  """Base error for this module."""


class TypeTransformationError(Error):
  """Error transforming between python proto type and corresponding C++ type."""


if _USE_C_DESCRIPTORS:
  # This metaclass allows to override the behavior of code like
  #     isinstance(my_descriptor, FieldDescriptor)
  # and make it return True when the descriptor is an instance of the extension
  # type written in C++.
  class DescriptorMetaclass(type):
    def __instancecheck__(cls, obj):
      if super(DescriptorMetaclass, cls).__instancecheck__(obj):
        return True
      if isinstance(obj, cls._C_DESCRIPTOR_CLASS):
        return True
      return False
else:
  # The standard metaclass; nothing changes.
  DescriptorMetaclass = type


class DescriptorBase(six.with_metaclass(DescriptorMetaclass)):

  """Descriptors base class.

  This class is the base of all descriptor classes. It provides common options
  related functionality.

  Attributes:
    has_options:  True if the descriptor has non-default options.  Usually it
        is not necessary to read this -- just call GetOptions() which will
        happily return the default instance.  However, it's sometimes useful
        for efficiency, and also useful inside the protobuf implementation to
        avoid some bootstrapping issues.
  """

  if _USE_C_DESCRIPTORS:
    # The class, or tuple of classes, that are considered as "virtual
    # subclasses" of this descriptor class.
    _C_DESCRIPTOR_CLASS = ()

  def __init__(self, options, options_class_name):
    """Initialize the descriptor given its options message and the name of the
    class of the options message. The name of the class is required in case
    the options message is None and has to be created.
    """
    self._options = options
    self._options_class_name = options_class_name

    # Does this descriptor have non-default options?
    self.has_options = options is not None

  def _SetOptions(self, options, options_class_name):
    """Sets the descriptor's options

    This function is used in generated proto2 files to update descriptor
    options. It must not be used outside proto2.
    """
    self._options = options
    self._options_class_name = options_class_name

    # Does this descriptor have non-default options?
    self.has_options = options is not None

  def GetOptions(self):
    """Retrieves descriptor options.

    This method returns the options set or creates the default options for the
    descriptor.
    """
    if self._options:
      return self._options
    from google.protobuf import descriptor_pb2
    try:
      options_class = getattr(descriptor_pb2, self._options_class_name)
    except AttributeError:
      raise RuntimeError('Unknown options class name %s!' %
                         (self._options_class_name))
    self._options = options_class()
    return self._options


class _NestedDescriptorBase(DescriptorBase):
  """Common class for descriptors that can be nested."""

  def __init__(self, options, options_class_name, name, full_name,
               file, containing_type, serialized_start=None,
               serialized_end=None):
    """Constructor.

    Args:
      options: Protocol message options or None
        to use default message options.
      options_class_name: (str) The class name of the above options.

      name: (str) Name of this protocol message type.
      full_name: (str) Fully-qualified name of this protocol message type,
        which will include protocol "package" name and the name of any
        enclosing types.
      file: (FileDescriptor) Reference to file info.
      containing_type: if provided, this is a nested descriptor, with this
        descriptor as parent, otherwise None.
      serialized_start: The start index (inclusive) in block in the
        file.serialized_pb that describes this descriptor.
      serialized_end: The end index (exclusive) in block in the
        file.serialized_pb that describes this descriptor.
    """
    super(_NestedDescriptorBase, self).__init__(
        options, options_class_name)

    self.name = name
    # TODO(falk): Add function to callwlate full_name instead of having it in
    #             memory?
    self.full_name = full_name
    self.file = file
    self.containing_type = containing_type

    self._serialized_start = serialized_start
    self._serialized_end = serialized_end

  def CopyToProto(self, proto):
    """Copies this to the matching proto in descriptor_pb2.

    Args:
      proto: An empty proto instance from descriptor_pb2.

    Raises:
      Error: If self couldnt be serialized, due to to few constructor arguments.
    """
    if (self.file is not None and
        self._serialized_start is not None and
        self._serialized_end is not None):
      proto.ParseFromString(self.file.serialized_pb[
          self._serialized_start:self._serialized_end])
    else:
      raise Error('Descriptor does not contain serialization.')


class Descriptor(_NestedDescriptorBase):

  """Descriptor for a protocol message type.

  A Descriptor instance has the following attributes:

    name: (str) Name of this protocol message type.
    full_name: (str) Fully-qualified name of this protocol message type,
      which will include protocol "package" name and the name of any
      enclosing types.

    containing_type: (Descriptor) Reference to the descriptor of the
      type containing us, or None if this is top-level.

    fields: (list of FieldDescriptors) Field descriptors for all
      fields in this type.
    fields_by_number: (dict int -> FieldDescriptor) Same FieldDescriptor
      objects as in |fields|, but indexed by "number" attribute in each
      FieldDescriptor.
    fields_by_name: (dict str -> FieldDescriptor) Same FieldDescriptor
      objects as in |fields|, but indexed by "name" attribute in each
      FieldDescriptor.
    fields_by_camelcase_name: (dict str -> FieldDescriptor) Same
      FieldDescriptor objects as in |fields|, but indexed by
      "camelcase_name" attribute in each FieldDescriptor.

    nested_types: (list of Descriptors) Descriptor references
      for all protocol message types nested within this one.
    nested_types_by_name: (dict str -> Descriptor) Same Descriptor
      objects as in |nested_types|, but indexed by "name" attribute
      in each Descriptor.

    enum_types: (list of EnumDescriptors) EnumDescriptor references
      for all enums contained within this type.
    enum_types_by_name: (dict str ->EnumDescriptor) Same EnumDescriptor
      objects as in |enum_types|, but indexed by "name" attribute
      in each EnumDescriptor.
    enum_values_by_name: (dict str -> EnumValueDescriptor) Dict mapping
      from enum value name to EnumValueDescriptor for that value.

    extensions: (list of FieldDescriptor) All extensions defined directly
      within this message type (NOT within a nested type).
    extensions_by_name: (dict, string -> FieldDescriptor) Same FieldDescriptor
      objects as |extensions|, but indexed by "name" attribute of each
      FieldDescriptor.

    is_extendable:  Does this type define any extension ranges?

    oneofs: (list of OneofDescriptor) The list of descriptors for oneof fields
      in this message.
    oneofs_by_name: (dict str -> OneofDescriptor) Same objects as in |oneofs|,
      but indexed by "name" attribute.

    file: (FileDescriptor) Reference to file descriptor.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.Descriptor

    def __new__(cls, name, full_name, filename, containing_type, fields,
                nested_types, enum_types, extensions, options=None,
                is_extendable=True, extension_ranges=None, oneofs=None,
                file=None, serialized_start=None, serialized_end=None,  # pylint: disable=redefined-builtin
                syntax=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindMessageTypeByName(full_name)

  # NOTE(tmarek): The file argument redefining a builtin is nothing we can
  # fix right now since we don't know how many clients already rely on the
  # name of the argument.
  def __init__(self, name, full_name, filename, containing_type, fields,
               nested_types, enum_types, extensions, options=None,
               is_extendable=True, extension_ranges=None, oneofs=None,
               file=None, serialized_start=None, serialized_end=None,  # pylint: disable=redefined-builtin
               syntax=None):
    """Arguments to __init__() are as described in the description
    of Descriptor fields above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
    super(Descriptor, self).__init__(
        options, 'MessageOptions', name, full_name, file,
        containing_type, serialized_start=serialized_start,
        serialized_end=serialized_end)

    # We have fields in addition to fields_by_name and fields_by_number,
    # so that:
    #   1. Clients can index fields by "order in which they're listed."
    #   2. Clients can easily iterate over all fields with the terse
    #      syntax: for f in descriptor.fields: ...
    self.fields = fields
    for field in self.fields:
      field.containing_type = self
    self.fields_by_number = dict((f.number, f) for f in fields)
    self.fields_by_name = dict((f.name, f) for f in fields)
    self._fields_by_camelcase_name = None

    self.nested_types = nested_types
    for nested_type in nested_types:
      nested_type.containing_type = self
    self.nested_types_by_name = dict((t.name, t) for t in nested_types)

    self.enum_types = enum_types
    for enum_type in self.enum_types:
      enum_type.containing_type = self
    self.enum_types_by_name = dict((t.name, t) for t in enum_types)
    self.enum_values_by_name = dict(
        (v.name, v) for t in enum_types for v in t.values)

    self.extensions = extensions
    for extension in self.extensions:
      extension.extension_scope = self
    self.extensions_by_name = dict((f.name, f) for f in extensions)
    self.is_extendable = is_extendable
    self.extension_ranges = extension_ranges
    self.oneofs = oneofs if oneofs is not None else []
    self.oneofs_by_name = dict((o.name, o) for o in self.oneofs)
    for oneof in self.oneofs:
      oneof.containing_type = self
    self.syntax = syntax or "proto2"

  @property
  def fields_by_camelcase_name(self):
    if self._fields_by_camelcase_name is None:
      self._fields_by_camelcase_name = dict(
          (f.camelcase_name, f) for f in self.fields)
    return self._fields_by_camelcase_name

  def EnumValueName(self, enum, value):
    """Returns the string name of an enum value.

    This is just a small helper method to simplify a common operation.

    Args:
      enum: string name of the Enum.
      value: int, value of the enum.

    Returns:
      string name of the enum value.

    Raises:
      KeyError if either the Enum doesn't exist or the value is not a valid
        value for the enum.
    """
    return self.enum_types_by_name[enum].values_by_number[value].name

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.DescriptorProto.

    Args:
      proto: An empty descriptor_pb2.DescriptorProto.
    """
    # This function is overridden to give a better doc comment.
    super(Descriptor, self).CopyToProto(proto)


# TODO(robinson): We should have aggressive checking here,
# for example:
#   * If you specify a repeated field, you should not be allowed
#     to specify a default value.
#   * [Other examples here as needed].
#
# TODO(robinson): for this and other *Descriptor classes, we
# might also want to lock things down aggressively (e.g.,
# prevent clients from setting the attributes).  Having
# stronger ilwariants here in general will reduce the number
# of runtime checks we must do in reflection.py...
class FieldDescriptor(DescriptorBase):

  """Descriptor for a single field in a .proto file.

  A FieldDescriptor instance has the following attributes:

    name: (str) Name of this field, exactly as it appears in .proto.
    full_name: (str) Name of this field, including containing scope.  This is
      partilwlarly relevant for extensions.
    camelcase_name: (str) Camelcase name of this field.
    index: (int) Dense, 0-indexed index giving the order that this
      field textually appears within its message in the .proto file.
    number: (int) Tag number declared for this field in the .proto file.

    type: (One of the TYPE_* constants below) Declared type.
    cpp_type: (One of the CPPTYPE_* constants below) C++ type used to
      represent this field.

    label: (One of the LABEL_* constants below) Tells whether this
      field is optional, required, or repeated.
    has_default_value: (bool) True if this field has a default value defined,
      otherwise false.
    default_value: (Varies) Default value of this field.  Only
      meaningful for non-repeated scalar fields.  Repeated fields
      should always set this to [], and non-repeated composite
      fields should always set this to None.

    containing_type: (Descriptor) Descriptor of the protocol message
      type that contains this field.  Set by the Descriptor constructor
      if we're passed into one.
      Somewhat confusingly, for extension fields, this is the
      descriptor of the EXTENDED message, not the descriptor
      of the message containing this field.  (See is_extension and
      extension_scope below).
    message_type: (Descriptor) If a composite field, a descriptor
      of the message type contained in this field.  Otherwise, this is None.
    enum_type: (EnumDescriptor) If this field contains an enum, a
      descriptor of that enum.  Otherwise, this is None.

    is_extension: True iff this describes an extension field.
    extension_scope: (Descriptor) Only meaningful if is_extension is True.
      Gives the message that immediately contains this extension field.
      Will be None iff we're a top-level (file-level) extension field.

    options: (descriptor_pb2.FieldOptions) Protocol message field options or
      None to use default field options.

    containing_oneof: (OneofDescriptor) If the field is a member of a oneof
      union, contains its descriptor. Otherwise, None.

    file: (FileDescriptor) Reference to file descriptor.
  """

  # Must be consistent with C++ FieldDescriptor::Type enum in
  # descriptor.h.
  #
  # TODO(robinson): Find a way to eliminate this repetition.
  TYPE_DOUBLE         = 1
  TYPE_FLOAT          = 2
  TYPE_INT64          = 3
  TYPE_UINT64         = 4
  TYPE_INT32          = 5
  TYPE_FIXED64        = 6
  TYPE_FIXED32        = 7
  TYPE_BOOL           = 8
  TYPE_STRING         = 9
  TYPE_GROUP          = 10
  TYPE_MESSAGE        = 11
  TYPE_BYTES          = 12
  TYPE_UINT32         = 13
  TYPE_ENUM           = 14
  TYPE_SFIXED32       = 15
  TYPE_SFIXED64       = 16
  TYPE_SINT32         = 17
  TYPE_SINT64         = 18
  MAX_TYPE            = 18

  # Must be consistent with C++ FieldDescriptor::CppType enum in
  # descriptor.h.
  #
  # TODO(robinson): Find a way to eliminate this repetition.
  CPPTYPE_INT32       = 1
  CPPTYPE_INT64       = 2
  CPPTYPE_UINT32      = 3
  CPPTYPE_UINT64      = 4
  CPPTYPE_DOUBLE      = 5
  CPPTYPE_FLOAT       = 6
  CPPTYPE_BOOL        = 7
  CPPTYPE_ENUM        = 8
  CPPTYPE_STRING      = 9
  CPPTYPE_MESSAGE     = 10
  MAX_CPPTYPE         = 10

  _PYTHON_TO_CPP_PROTO_TYPE_MAP = {
      TYPE_DOUBLE: CPPTYPE_DOUBLE,
      TYPE_FLOAT: CPPTYPE_FLOAT,
      TYPE_ENUM: CPPTYPE_ENUM,
      TYPE_INT64: CPPTYPE_INT64,
      TYPE_SINT64: CPPTYPE_INT64,
      TYPE_SFIXED64: CPPTYPE_INT64,
      TYPE_UINT64: CPPTYPE_UINT64,
      TYPE_FIXED64: CPPTYPE_UINT64,
      TYPE_INT32: CPPTYPE_INT32,
      TYPE_SFIXED32: CPPTYPE_INT32,
      TYPE_SINT32: CPPTYPE_INT32,
      TYPE_UINT32: CPPTYPE_UINT32,
      TYPE_FIXED32: CPPTYPE_UINT32,
      TYPE_BYTES: CPPTYPE_STRING,
      TYPE_STRING: CPPTYPE_STRING,
      TYPE_BOOL: CPPTYPE_BOOL,
      TYPE_MESSAGE: CPPTYPE_MESSAGE,
      TYPE_GROUP: CPPTYPE_MESSAGE
      }

  # Must be consistent with C++ FieldDescriptor::Label enum in
  # descriptor.h.
  #
  # TODO(robinson): Find a way to eliminate this repetition.
  LABEL_OPTIONAL      = 1
  LABEL_REQUIRED      = 2
  LABEL_REPEATED      = 3
  MAX_LABEL           = 3

  # Must be consistent with C++ constants kMaxNumber, kFirstReservedNumber,
  # and kLastReservedNumber in descriptor.h
  MAX_FIELD_NUMBER = (1 << 29) - 1
  FIRST_RESERVED_FIELD_NUMBER = 19000
  LAST_RESERVED_FIELD_NUMBER = 19999

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.FieldDescriptor

    def __new__(cls, name, full_name, index, number, type, cpp_type, label,
                default_value, message_type, enum_type, containing_type,
                is_extension, extension_scope, options=None,
                has_default_value=True, containing_oneof=None, json_name=None,
                file=None):
      _message.Message._CheckCalledFromGeneratedFile()
      if is_extension:
        return _message.default_pool.FindExtensionByName(full_name)
      else:
        return _message.default_pool.FindFieldByName(full_name)

  def __init__(self, name, full_name, index, number, type, cpp_type, label,
               default_value, message_type, enum_type, containing_type,
               is_extension, extension_scope, options=None,
               has_default_value=True, containing_oneof=None, json_name=None,
               file=None):
    """The arguments are as described in the description of FieldDescriptor
    attributes above.

    Note that containing_type may be None, and may be set later if necessary
    (to deal with cirlwlar references between message types, for example).
    Likewise for extension_scope.
    """
    super(FieldDescriptor, self).__init__(options, 'FieldOptions')
    self.name = name
    self.full_name = full_name
    self.file = file
    self._camelcase_name = None
    if json_name is None:
      self.json_name = _ToJsonName(name)
    else:
      self.json_name = json_name
    self.index = index
    self.number = number
    self.type = type
    self.cpp_type = cpp_type
    self.label = label
    self.has_default_value = has_default_value
    self.default_value = default_value
    self.containing_type = containing_type
    self.message_type = message_type
    self.enum_type = enum_type
    self.is_extension = is_extension
    self.extension_scope = extension_scope
    self.containing_oneof = containing_oneof
    if api_implementation.Type() == 'cpp':
      if is_extension:
        self._cdescriptor = _message.default_pool.FindExtensionByName(full_name)
      else:
        self._cdescriptor = _message.default_pool.FindFieldByName(full_name)
    else:
      self._cdescriptor = None

  @property
  def camelcase_name(self):
    if self._camelcase_name is None:
      self._camelcase_name = _ToCamelCase(self.name)
    return self._camelcase_name

  @staticmethod
  def ProtoTypeToCppProtoType(proto_type):
    """Colwerts from a Python proto type to a C++ Proto Type.

    The Python ProtocolBuffer classes specify both the 'Python' datatype and the
    'C++' datatype - and they're not the same. This helper method should
    translate from one to another.

    Args:
      proto_type: the Python proto type (descriptor.FieldDescriptor.TYPE_*)
    Returns:
      descriptor.FieldDescriptor.CPPTYPE_*, the C++ type.
    Raises:
      TypeTransformationError: when the Python proto type isn't known.
    """
    try:
      return FieldDescriptor._PYTHON_TO_CPP_PROTO_TYPE_MAP[proto_type]
    except KeyError:
      raise TypeTransformationError('Unknown proto_type: %s' % proto_type)


class EnumDescriptor(_NestedDescriptorBase):

  """Descriptor for an enum defined in a .proto file.

  An EnumDescriptor instance has the following attributes:

    name: (str) Name of the enum type.
    full_name: (str) Full name of the type, including package name
      and any enclosing type(s).

    values: (list of EnumValueDescriptors) List of the values
      in this enum.
    values_by_name: (dict str -> EnumValueDescriptor) Same as |values|,
      but indexed by the "name" field of each EnumValueDescriptor.
    values_by_number: (dict int -> EnumValueDescriptor) Same as |values|,
      but indexed by the "number" field of each EnumValueDescriptor.
    containing_type: (Descriptor) Descriptor of the immediate containing
      type of this enum, or None if this is an enum defined at the
      top level in a .proto file.  Set by Descriptor's constructor
      if we're passed into one.
    file: (FileDescriptor) Reference to file descriptor.
    options: (descriptor_pb2.EnumOptions) Enum options message or
      None to use default enum options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.EnumDescriptor

    def __new__(cls, name, full_name, filename, values,
                containing_type=None, options=None, file=None,
                serialized_start=None, serialized_end=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindEnumTypeByName(full_name)

  def __init__(self, name, full_name, filename, values,
               containing_type=None, options=None, file=None,
               serialized_start=None, serialized_end=None):
    """Arguments are as described in the attribute description above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
    super(EnumDescriptor, self).__init__(
        options, 'EnumOptions', name, full_name, file,
        containing_type, serialized_start=serialized_start,
        serialized_end=serialized_end)

    self.values = values
    for value in self.values:
      value.type = self
    self.values_by_name = dict((v.name, v) for v in values)
    self.values_by_number = dict((v.number, v) for v in values)

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.EnumDescriptorProto.

    Args:
      proto: An empty descriptor_pb2.EnumDescriptorProto.
    """
    # This function is overridden to give a better doc comment.
    super(EnumDescriptor, self).CopyToProto(proto)


class EnumValueDescriptor(DescriptorBase):

  """Descriptor for a single value within an enum.

    name: (str) Name of this value.
    index: (int) Dense, 0-indexed index giving the order that this
      value appears textually within its enum in the .proto file.
    number: (int) Actual number assigned to this enum value.
    type: (EnumDescriptor) EnumDescriptor to which this value
      belongs.  Set by EnumDescriptor's constructor if we're
      passed into one.
    options: (descriptor_pb2.EnumValueOptions) Enum value options message or
      None to use default enum value options options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.EnumValueDescriptor

    def __new__(cls, name, index, number, type=None, options=None):
      _message.Message._CheckCalledFromGeneratedFile()
      # There is no way we can build a complete EnumValueDescriptor with the
      # given parameters (the name of the Enum is not known, for example).
      # Fortunately generated files just pass it to the EnumDescriptor()
      # constructor, which will ignore it, so returning None is good enough.
      return None

  def __init__(self, name, index, number, type=None, options=None):
    """Arguments are as described in the attribute description above."""
    super(EnumValueDescriptor, self).__init__(options, 'EnumValueOptions')
    self.name = name
    self.index = index
    self.number = number
    self.type = type


class OneofDescriptor(DescriptorBase):
  """Descriptor for a oneof field.

    name: (str) Name of the oneof field.
    full_name: (str) Full name of the oneof field, including package name.
    index: (int) 0-based index giving the order of the oneof field inside
      its containing type.
    containing_type: (Descriptor) Descriptor of the protocol message
      type that contains this field.  Set by the Descriptor constructor
      if we're passed into one.
    fields: (list of FieldDescriptor) The list of field descriptors this
      oneof can contain.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.OneofDescriptor

    def __new__(
        cls, name, full_name, index, containing_type, fields, options=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindOneofByName(full_name)

  def __init__(
      self, name, full_name, index, containing_type, fields, options=None):
    """Arguments are as described in the attribute description above."""
    super(OneofDescriptor, self).__init__(options, 'OneofOptions')
    self.name = name
    self.full_name = full_name
    self.index = index
    self.containing_type = containing_type
    self.fields = fields


class ServiceDescriptor(_NestedDescriptorBase):

  """Descriptor for a service.

    name: (str) Name of the service.
    full_name: (str) Full name of the service, including package name.
    index: (int) 0-indexed index giving the order that this services
      definition appears withing the .proto file.
    methods: (list of MethodDescriptor) List of methods provided by this
      service.
    methods_by_name: (dict str -> MethodDescriptor) Same MethodDescriptor
      objects as in |methods_by_name|, but indexed by "name" attribute in each
      MethodDescriptor.
    options: (descriptor_pb2.ServiceOptions) Service options message or
      None to use default service options.
    file: (FileDescriptor) Reference to file info.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.ServiceDescriptor

    def __new__(cls, name, full_name, index, methods, options=None, file=None,  # pylint: disable=redefined-builtin
                serialized_start=None, serialized_end=None):
      _message.Message._CheckCalledFromGeneratedFile()  # pylint: disable=protected-access
      return _message.default_pool.FindServiceByName(full_name)

  def __init__(self, name, full_name, index, methods, options=None, file=None,
               serialized_start=None, serialized_end=None):
    super(ServiceDescriptor, self).__init__(
        options, 'ServiceOptions', name, full_name, file,
        None, serialized_start=serialized_start,
        serialized_end=serialized_end)
    self.index = index
    self.methods = methods
    self.methods_by_name = dict((m.name, m) for m in methods)
    # Set the containing service for each method in this service.
    for method in self.methods:
      method.containing_service = self

  def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor."""
    return self.methods_by_name.get(name, None)

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.ServiceDescriptorProto.

    Args:
      proto: An empty descriptor_pb2.ServiceDescriptorProto.
    """
    # This function is overridden to give a better doc comment.
    super(ServiceDescriptor, self).CopyToProto(proto)


class MethodDescriptor(DescriptorBase):

  """Descriptor for a method in a service.

  name: (str) Name of the method within the service.
  full_name: (str) Full name of method.
  index: (int) 0-indexed index of the method inside the service.
  containing_service: (ServiceDescriptor) The service that contains this
    method.
  input_type: The descriptor of the message that this method accepts.
  output_type: The descriptor of the message that this method returns.
  options: (descriptor_pb2.MethodOptions) Method options message or
    None to use default method options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.MethodDescriptor

    def __new__(cls, name, full_name, index, containing_service,
                input_type, output_type, options=None):
      _message.Message._CheckCalledFromGeneratedFile()  # pylint: disable=protected-access
      return _message.default_pool.FindMethodByName(full_name)

  def __init__(self, name, full_name, index, containing_service,
               input_type, output_type, options=None):
    """The arguments are as described in the description of MethodDescriptor
    attributes above.

    Note that containing_service may be None, and may be set later if necessary.
    """
    super(MethodDescriptor, self).__init__(options, 'MethodOptions')
    self.name = name
    self.full_name = full_name
    self.index = index
    self.containing_service = containing_service
    self.input_type = input_type
    self.output_type = output_type


class FileDescriptor(DescriptorBase):
  """Descriptor for a file. Mimics the descriptor_pb2.FileDescriptorProto.

  Note that enum_types_by_name, extensions_by_name, and dependencies
  fields are only set by the message_factory module, and not by the
  generated proto code.

  name: name of file, relative to root of source tree.
  package: name of the package
  syntax: string indicating syntax of the file (can be "proto2" or "proto3")
  serialized_pb: (str) Byte string of serialized
    descriptor_pb2.FileDescriptorProto.
  dependencies: List of other FileDescriptors this FileDescriptor depends on.
  public_dependencies: A list of FileDescriptors, subset of the dependencies
    above, which were declared as "public".
  message_types_by_name: Dict of message names of their descriptors.
  enum_types_by_name: Dict of enum names and their descriptors.
  extensions_by_name: Dict of extension names and their descriptors.
  services_by_name: Dict of services names and their descriptors.
  pool: the DescriptorPool this descriptor belongs to.  When not passed to the
    constructor, the global default pool is used.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.FileDescriptor

    def __new__(cls, name, package, options=None, serialized_pb=None,
                dependencies=None, public_dependencies=None,
                syntax=None, pool=None):
      # FileDescriptor() is called from various places, not only from generated
      # files, to register dynamic proto files and messages.
      if serialized_pb:
        # TODO(amauryfa): use the pool passed as argument. This will work only
        # for C++-implemented DescriptorPools.
        return _message.default_pool.AddSerializedFile(serialized_pb)
      else:
        return super(FileDescriptor, cls).__new__(cls)

  def __init__(self, name, package, options=None, serialized_pb=None,
               dependencies=None, public_dependencies=None,
               syntax=None, pool=None):
    """Constructor."""
    super(FileDescriptor, self).__init__(options, 'FileOptions')

    if pool is None:
      from google.protobuf import descriptor_pool
      pool = descriptor_pool.Default()
    self.pool = pool
    self.message_types_by_name = {}
    self.name = name
    self.package = package
    self.syntax = syntax or "proto2"
    self.serialized_pb = serialized_pb

    self.enum_types_by_name = {}
    self.extensions_by_name = {}
    self.services_by_name = {}
    self.dependencies = (dependencies or [])
    self.public_dependencies = (public_dependencies or [])

    if (api_implementation.Type() == 'cpp' and
        self.serialized_pb is not None):
      _message.default_pool.AddSerializedFile(self.serialized_pb)

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.FileDescriptorProto.

    Args:
      proto: An empty descriptor_pb2.FileDescriptorProto.
    """
    proto.ParseFromString(self.serialized_pb)


def _ParseOptions(message, string):
  """Parses serialized options.

  This helper function is used to parse serialized options in generated
  proto2 files. It must not be used outside proto2.
  """
  message.ParseFromString(string)
  return message


def _ToCamelCase(name):
  """Colwerts name to camel-case and returns it."""
  capitalize_next = False
  result = []

  for c in name:
    if c == '_':
      if result:
        capitalize_next = True
    elif capitalize_next:
      result.append(c.upper())
      capitalize_next = False
    else:
      result += c

  # Lower-case the first letter.
  if result and result[0].isupper():
    result[0] = result[0].lower()
  return ''.join(result)


def _OptionsOrNone(descriptor_proto):
  """Returns the value of the field `options`, or None if it is not set."""
  if descriptor_proto.HasField('options'):
    return descriptor_proto.options
  else:
    return None


def _ToJsonName(name):
  """Colwerts name to Json name and returns it."""
  capitalize_next = False
  result = []

  for c in name:
    if c == '_':
      capitalize_next = True
    elif capitalize_next:
      result.append(c.upper())
      capitalize_next = False
    else:
      result += c

  return ''.join(result)


def MakeDescriptor(desc_proto, package='', build_file_if_cpp=True,
                   syntax=None):
  """Make a protobuf Descriptor given a DescriptorProto protobuf.

  Handles nested descriptors. Note that this is limited to the scope of defining
  a message inside of another message. Composite fields can lwrrently only be
  resolved if the message is defined in the same scope as the field.

  Args:
    desc_proto: The descriptor_pb2.DescriptorProto protobuf message.
    package: Optional package name for the new message Descriptor (string).
    build_file_if_cpp: Update the C++ descriptor pool if api matches.
                       Set to False on relwrsion, so no duplicates are created.
    syntax: The syntax/semantics that should be used.  Set to "proto3" to get
            proto3 field presence semantics.
  Returns:
    A Descriptor for protobuf messages.
  """
  if api_implementation.Type() == 'cpp' and build_file_if_cpp:
    # The C++ implementation requires all descriptors to be backed by the same
    # definition in the C++ descriptor pool. To do this, we build a
    # FileDescriptorProto with the same definition as this descriptor and build
    # it into the pool.
    from google.protobuf import descriptor_pb2
    file_descriptor_proto = descriptor_pb2.FileDescriptorProto()
    file_descriptor_proto.message_type.add().MergeFrom(desc_proto)

    # Generate a random name for this proto file to prevent conflicts with any
    # imported ones. We need to specify a file name so the descriptor pool
    # accepts our FileDescriptorProto, but it is not important what that file
    # name is actually set to.
    proto_name = str(uuid.uuid4())

    if package:
      file_descriptor_proto.name = os.path.join(package.replace('.', '/'),
                                                proto_name + '.proto')
      file_descriptor_proto.package = package
    else:
      file_descriptor_proto.name = proto_name + '.proto'

    _message.default_pool.Add(file_descriptor_proto)
    result = _message.default_pool.FindFileByName(file_descriptor_proto.name)

    if _USE_C_DESCRIPTORS:
      return result.message_types_by_name[desc_proto.name]

  full_message_name = [desc_proto.name]
  if package: full_message_name.insert(0, package)

  # Create Descriptors for enum types
  enum_types = {}
  for enum_proto in desc_proto.enum_type:
    full_name = '.'.join(full_message_name + [enum_proto.name])
    enum_desc = EnumDescriptor(
      enum_proto.name, full_name, None, [
          EnumValueDescriptor(enum_val.name, ii, enum_val.number)
          for ii, enum_val in enumerate(enum_proto.value)])
    enum_types[full_name] = enum_desc

  # Create Descriptors for nested types
  nested_types = {}
  for nested_proto in desc_proto.nested_type:
    full_name = '.'.join(full_message_name + [nested_proto.name])
    # Nested types are just those defined inside of the message, not all types
    # used by fields in the message, so no loops are possible here.
    nested_desc = MakeDescriptor(nested_proto,
                                 package='.'.join(full_message_name),
                                 build_file_if_cpp=False,
                                 syntax=syntax)
    nested_types[full_name] = nested_desc

  fields = []
  for field_proto in desc_proto.field:
    full_name = '.'.join(full_message_name + [field_proto.name])
    enum_desc = None
    nested_desc = None
    if field_proto.json_name:
      json_name = field_proto.json_name
    else:
      json_name = None
    if field_proto.HasField('type_name'):
      type_name = field_proto.type_name
      full_type_name = '.'.join(full_message_name +
                                [type_name[type_name.rfind('.')+1:]])
      if full_type_name in nested_types:
        nested_desc = nested_types[full_type_name]
      elif full_type_name in enum_types:
        enum_desc = enum_types[full_type_name]
      # Else type_name references a non-local type, which isn't implemented
    field = FieldDescriptor(
        field_proto.name, full_name, field_proto.number - 1,
        field_proto.number, field_proto.type,
        FieldDescriptor.ProtoTypeToCppProtoType(field_proto.type),
        field_proto.label, None, nested_desc, enum_desc, None, False, None,
        options=_OptionsOrNone(field_proto), has_default_value=False,
        json_name=json_name)
    fields.append(field)

  desc_name = '.'.join(full_message_name)
  return Descriptor(desc_proto.name, desc_name, None, None, fields,
                    list(nested_types.values()), list(enum_types.values()), [],
                    options=_OptionsOrNone(desc_proto))
