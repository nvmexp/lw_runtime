# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/protobuf/internal/more_extensions_dynamic.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf.internal import more_extensions_pb2 as google_dot_protobuf_dot_internal_dot_more__extensions__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='google/protobuf/internal/more_extensions_dynamic.proto',
  package='google.protobuf.internal',
  syntax='proto2',
  serialized_pb=_b('\n6google/protobuf/internal/more_extensions_dynamic.proto\x12\x18google.protobuf.internal\x1a.google/protobuf/internal/more_extensions.proto\"\x1f\n\x12\x44ynamicMessageType\x12\t\n\x01\x61\x18\x01 \x01(\x05:J\n\x17\x64ynamic_int32_extension\x12).google.protobuf.internal.ExtendedMessage\x18\x64 \x01(\x05:z\n\x19\x64ynamic_message_extension\x12).google.protobuf.internal.ExtendedMessage\x18\x65 \x01(\x0b\x32,.google.protobuf.internal.DynamicMessageType:\x83\x01\n\"repeated_dynamic_message_extension\x12).google.protobuf.internal.ExtendedMessage\x18\x66 \x03(\x0b\x32,.google.protobuf.internal.DynamicMessageType')
  ,
  dependencies=[google_dot_protobuf_dot_internal_dot_more__extensions__pb2.DESCRIPTOR,])


DYNAMIC_INT32_EXTENSION_FIELD_NUMBER = 100
dynamic_int32_extension = _descriptor.FieldDescriptor(
  name='dynamic_int32_extension', full_name='google.protobuf.internal.dynamic_int32_extension', index=0,
  number=100, type=5, cpp_type=1, label=1,
  has_default_value=False, default_value=0,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None, file=DESCRIPTOR)
DYNAMIC_MESSAGE_EXTENSION_FIELD_NUMBER = 101
dynamic_message_extension = _descriptor.FieldDescriptor(
  name='dynamic_message_extension', full_name='google.protobuf.internal.dynamic_message_extension', index=1,
  number=101, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None, file=DESCRIPTOR)
REPEATED_DYNAMIC_MESSAGE_EXTENSION_FIELD_NUMBER = 102
repeated_dynamic_message_extension = _descriptor.FieldDescriptor(
  name='repeated_dynamic_message_extension', full_name='google.protobuf.internal.repeated_dynamic_message_extension', index=2,
  number=102, type=11, cpp_type=10, label=3,
  has_default_value=False, default_value=[],
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None, file=DESCRIPTOR)


_DYNAMICMESSAGETYPE = _descriptor.Descriptor(
  name='DynamicMessageType',
  full_name='google.protobuf.internal.DynamicMessageType',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='a', full_name='google.protobuf.internal.DynamicMessageType.a', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=132,
  serialized_end=163,
)

DESCRIPTOR.message_types_by_name['DynamicMessageType'] = _DYNAMICMESSAGETYPE
DESCRIPTOR.extensions_by_name['dynamic_int32_extension'] = dynamic_int32_extension
DESCRIPTOR.extensions_by_name['dynamic_message_extension'] = dynamic_message_extension
DESCRIPTOR.extensions_by_name['repeated_dynamic_message_extension'] = repeated_dynamic_message_extension
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DynamicMessageType = _reflection.GeneratedProtocolMessageType('DynamicMessageType', (_message.Message,), dict(
  DESCRIPTOR = _DYNAMICMESSAGETYPE,
  __module__ = 'google.protobuf.internal.more_extensions_dynamic_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.internal.DynamicMessageType)
  ))
_sym_db.RegisterMessage(DynamicMessageType)

google_dot_protobuf_dot_internal_dot_more__extensions__pb2.ExtendedMessage.RegisterExtension(dynamic_int32_extension)
dynamic_message_extension.message_type = _DYNAMICMESSAGETYPE
google_dot_protobuf_dot_internal_dot_more__extensions__pb2.ExtendedMessage.RegisterExtension(dynamic_message_extension)
repeated_dynamic_message_extension.message_type = _DYNAMICMESSAGETYPE
google_dot_protobuf_dot_internal_dot_more__extensions__pb2.ExtendedMessage.RegisterExtension(repeated_dynamic_message_extension)

# @@protoc_insertion_point(module_scope)
