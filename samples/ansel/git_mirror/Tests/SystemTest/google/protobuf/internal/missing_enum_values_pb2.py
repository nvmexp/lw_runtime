# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/protobuf/internal/missing_enum_values.proto

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
  name='google/protobuf/internal/missing_enum_values.proto',
  package='google.protobuf.python.internal',
  syntax='proto2',
  serialized_pb=_b('\n2google/protobuf/internal/missing_enum_values.proto\x12\x1fgoogle.protobuf.python.internal\"\xc1\x02\n\x0eTestEnumValues\x12X\n\x14optional_nested_enum\x18\x01 \x01(\x0e\x32:.google.protobuf.python.internal.TestEnumValues.NestedEnum\x12X\n\x14repeated_nested_enum\x18\x02 \x03(\x0e\x32:.google.protobuf.python.internal.TestEnumValues.NestedEnum\x12Z\n\x12packed_nested_enum\x18\x03 \x03(\x0e\x32:.google.protobuf.python.internal.TestEnumValues.NestedEnumB\x02\x10\x01\"\x1f\n\nNestedEnum\x12\x08\n\x04ZERO\x10\x00\x12\x07\n\x03ONE\x10\x01\"\xd3\x02\n\x15TestMissingEnumValues\x12_\n\x14optional_nested_enum\x18\x01 \x01(\x0e\x32\x41.google.protobuf.python.internal.TestMissingEnumValues.NestedEnum\x12_\n\x14repeated_nested_enum\x18\x02 \x03(\x0e\x32\x41.google.protobuf.python.internal.TestMissingEnumValues.NestedEnum\x12\x61\n\x12packed_nested_enum\x18\x03 \x03(\x0e\x32\x41.google.protobuf.python.internal.TestMissingEnumValues.NestedEnumB\x02\x10\x01\"\x15\n\nNestedEnum\x12\x07\n\x03TWO\x10\x02\"\x1b\n\nJustString\x12\r\n\x05\x64ummy\x18\x01 \x02(\t')
)



_TESTENUMVALUES_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.TestEnumValues.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ONE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=378,
  serialized_end=409,
)
_sym_db.RegisterEnumDescriptor(_TESTENUMVALUES_NESTEDENUM)

_TESTMISSINGENUMVALUES_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.TestMissingEnumValues.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TWO', index=0, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=730,
  serialized_end=751,
)
_sym_db.RegisterEnumDescriptor(_TESTMISSINGENUMVALUES_NESTEDENUM)


_TESTENUMVALUES = _descriptor.Descriptor(
  name='TestEnumValues',
  full_name='google.protobuf.python.internal.TestEnumValues',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='optional_nested_enum', full_name='google.protobuf.python.internal.TestEnumValues.optional_nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='repeated_nested_enum', full_name='google.protobuf.python.internal.TestEnumValues.repeated_nested_enum', index=1,
      number=2, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='packed_nested_enum', full_name='google.protobuf.python.internal.TestEnumValues.packed_nested_enum', index=2,
      number=3, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001')), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TESTENUMVALUES_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=88,
  serialized_end=409,
)


_TESTMISSINGENUMVALUES = _descriptor.Descriptor(
  name='TestMissingEnumValues',
  full_name='google.protobuf.python.internal.TestMissingEnumValues',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='optional_nested_enum', full_name='google.protobuf.python.internal.TestMissingEnumValues.optional_nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='repeated_nested_enum', full_name='google.protobuf.python.internal.TestMissingEnumValues.repeated_nested_enum', index=1,
      number=2, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='packed_nested_enum', full_name='google.protobuf.python.internal.TestMissingEnumValues.packed_nested_enum', index=2,
      number=3, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001')), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TESTMISSINGENUMVALUES_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=412,
  serialized_end=751,
)


_JUSTSTRING = _descriptor.Descriptor(
  name='JustString',
  full_name='google.protobuf.python.internal.JustString',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dummy', full_name='google.protobuf.python.internal.JustString.dummy', index=0,
      number=1, type=9, cpp_type=9, label=2,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=753,
  serialized_end=780,
)

_TESTENUMVALUES.fields_by_name['optional_nested_enum'].enum_type = _TESTENUMVALUES_NESTEDENUM
_TESTENUMVALUES.fields_by_name['repeated_nested_enum'].enum_type = _TESTENUMVALUES_NESTEDENUM
_TESTENUMVALUES.fields_by_name['packed_nested_enum'].enum_type = _TESTENUMVALUES_NESTEDENUM
_TESTENUMVALUES_NESTEDENUM.containing_type = _TESTENUMVALUES
_TESTMISSINGENUMVALUES.fields_by_name['optional_nested_enum'].enum_type = _TESTMISSINGENUMVALUES_NESTEDENUM
_TESTMISSINGENUMVALUES.fields_by_name['repeated_nested_enum'].enum_type = _TESTMISSINGENUMVALUES_NESTEDENUM
_TESTMISSINGENUMVALUES.fields_by_name['packed_nested_enum'].enum_type = _TESTMISSINGENUMVALUES_NESTEDENUM
_TESTMISSINGENUMVALUES_NESTEDENUM.containing_type = _TESTMISSINGENUMVALUES
DESCRIPTOR.message_types_by_name['TestEnumValues'] = _TESTENUMVALUES
DESCRIPTOR.message_types_by_name['TestMissingEnumValues'] = _TESTMISSINGENUMVALUES
DESCRIPTOR.message_types_by_name['JustString'] = _JUSTSTRING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TestEnumValues = _reflection.GeneratedProtocolMessageType('TestEnumValues', (_message.Message,), dict(
  DESCRIPTOR = _TESTENUMVALUES,
  __module__ = 'google.protobuf.internal.missing_enum_values_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.TestEnumValues)
  ))
_sym_db.RegisterMessage(TestEnumValues)

TestMissingEnumValues = _reflection.GeneratedProtocolMessageType('TestMissingEnumValues', (_message.Message,), dict(
  DESCRIPTOR = _TESTMISSINGENUMVALUES,
  __module__ = 'google.protobuf.internal.missing_enum_values_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.TestMissingEnumValues)
  ))
_sym_db.RegisterMessage(TestMissingEnumValues)

JustString = _reflection.GeneratedProtocolMessageType('JustString', (_message.Message,), dict(
  DESCRIPTOR = _JUSTSTRING,
  __module__ = 'google.protobuf.internal.missing_enum_values_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.JustString)
  ))
_sym_db.RegisterMessage(JustString)


_TESTENUMVALUES.fields_by_name['packed_nested_enum'].has_options = True
_TESTENUMVALUES.fields_by_name['packed_nested_enum']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_TESTMISSINGENUMVALUES.fields_by_name['packed_nested_enum'].has_options = True
_TESTMISSINGENUMVALUES.fields_by_name['packed_nested_enum']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
