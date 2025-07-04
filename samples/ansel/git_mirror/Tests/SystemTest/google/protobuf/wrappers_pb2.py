# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/protobuf/wrappers.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='google/protobuf/wrappers.proto',
  package='google.protobuf',
  syntax='proto3',
  serialized_pb=_b('\n\x1egoogle/protobuf/wrappers.proto\x12\x0fgoogle.protobuf\"\x1c\n\x0b\x44oubleValue\x12\r\n\x05value\x18\x01 \x01(\x01\"\x1b\n\nFloatValue\x12\r\n\x05value\x18\x01 \x01(\x02\"\x1b\n\nInt64Value\x12\r\n\x05value\x18\x01 \x01(\x03\"\x1c\n\x0bUInt64Value\x12\r\n\x05value\x18\x01 \x01(\x04\"\x1b\n\nInt32Value\x12\r\n\x05value\x18\x01 \x01(\x05\"\x1c\n\x0bUInt32Value\x12\r\n\x05value\x18\x01 \x01(\r\"\x1a\n\tBoolValue\x12\r\n\x05value\x18\x01 \x01(\x08\"\x1c\n\x0bStringValue\x12\r\n\x05value\x18\x01 \x01(\t\"\x1b\n\nBytesValue\x12\r\n\x05value\x18\x01 \x01(\x0c\x42|\n\x13\x63om.google.protobufB\rWrappersProtoP\x01Z*github.com/golang/protobuf/ptypes/wrappers\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')
)




_DOUBLEVALUE = _descriptor.Descriptor(
  name='DoubleValue',
  full_name='google.protobuf.DoubleValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.DoubleValue.value', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=79,
)


_FLOATVALUE = _descriptor.Descriptor(
  name='FloatValue',
  full_name='google.protobuf.FloatValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.FloatValue.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=81,
  serialized_end=108,
)


_INT64VALUE = _descriptor.Descriptor(
  name='Int64Value',
  full_name='google.protobuf.Int64Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.Int64Value.value', index=0,
      number=1, type=3, cpp_type=2, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=137,
)


_UINT64VALUE = _descriptor.Descriptor(
  name='UInt64Value',
  full_name='google.protobuf.UInt64Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.UInt64Value.value', index=0,
      number=1, type=4, cpp_type=4, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=139,
  serialized_end=167,
)


_INT32VALUE = _descriptor.Descriptor(
  name='Int32Value',
  full_name='google.protobuf.Int32Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.Int32Value.value', index=0,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=169,
  serialized_end=196,
)


_UINT32VALUE = _descriptor.Descriptor(
  name='UInt32Value',
  full_name='google.protobuf.UInt32Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.UInt32Value.value', index=0,
      number=1, type=13, cpp_type=3, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=198,
  serialized_end=226,
)


_BOOLVALUE = _descriptor.Descriptor(
  name='BoolValue',
  full_name='google.protobuf.BoolValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.BoolValue.value', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=228,
  serialized_end=254,
)


_STRINGVALUE = _descriptor.Descriptor(
  name='StringValue',
  full_name='google.protobuf.StringValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.StringValue.value', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=256,
  serialized_end=284,
)


_BYTESVALUE = _descriptor.Descriptor(
  name='BytesValue',
  full_name='google.protobuf.BytesValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='google.protobuf.BytesValue.value', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=286,
  serialized_end=313,
)

DESCRIPTOR.message_types_by_name['DoubleValue'] = _DOUBLEVALUE
DESCRIPTOR.message_types_by_name['FloatValue'] = _FLOATVALUE
DESCRIPTOR.message_types_by_name['Int64Value'] = _INT64VALUE
DESCRIPTOR.message_types_by_name['UInt64Value'] = _UINT64VALUE
DESCRIPTOR.message_types_by_name['Int32Value'] = _INT32VALUE
DESCRIPTOR.message_types_by_name['UInt32Value'] = _UINT32VALUE
DESCRIPTOR.message_types_by_name['BoolValue'] = _BOOLVALUE
DESCRIPTOR.message_types_by_name['StringValue'] = _STRINGVALUE
DESCRIPTOR.message_types_by_name['BytesValue'] = _BYTESVALUE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DoubleValue = _reflection.GeneratedProtocolMessageType('DoubleValue', (_message.Message,), dict(
  DESCRIPTOR = _DOUBLEVALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.DoubleValue)
  ))
_sym_db.RegisterMessage(DoubleValue)

FloatValue = _reflection.GeneratedProtocolMessageType('FloatValue', (_message.Message,), dict(
  DESCRIPTOR = _FLOATVALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.FloatValue)
  ))
_sym_db.RegisterMessage(FloatValue)

Int64Value = _reflection.GeneratedProtocolMessageType('Int64Value', (_message.Message,), dict(
  DESCRIPTOR = _INT64VALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.Int64Value)
  ))
_sym_db.RegisterMessage(Int64Value)

UInt64Value = _reflection.GeneratedProtocolMessageType('UInt64Value', (_message.Message,), dict(
  DESCRIPTOR = _UINT64VALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.UInt64Value)
  ))
_sym_db.RegisterMessage(UInt64Value)

Int32Value = _reflection.GeneratedProtocolMessageType('Int32Value', (_message.Message,), dict(
  DESCRIPTOR = _INT32VALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.Int32Value)
  ))
_sym_db.RegisterMessage(Int32Value)

UInt32Value = _reflection.GeneratedProtocolMessageType('UInt32Value', (_message.Message,), dict(
  DESCRIPTOR = _UINT32VALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.UInt32Value)
  ))
_sym_db.RegisterMessage(UInt32Value)

BoolValue = _reflection.GeneratedProtocolMessageType('BoolValue', (_message.Message,), dict(
  DESCRIPTOR = _BOOLVALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.BoolValue)
  ))
_sym_db.RegisterMessage(BoolValue)

StringValue = _reflection.GeneratedProtocolMessageType('StringValue', (_message.Message,), dict(
  DESCRIPTOR = _STRINGVALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.StringValue)
  ))
_sym_db.RegisterMessage(StringValue)

BytesValue = _reflection.GeneratedProtocolMessageType('BytesValue', (_message.Message,), dict(
  DESCRIPTOR = _BYTESVALUE,
  __module__ = 'google.protobuf.wrappers_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.BytesValue)
  ))
_sym_db.RegisterMessage(BytesValue)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\023com.google.protobufB\rWrappersProtoP\001Z*github.com/golang/protobuf/ptypes/wrappers\370\001\001\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'))
# @@protoc_insertion_point(module_scope)
