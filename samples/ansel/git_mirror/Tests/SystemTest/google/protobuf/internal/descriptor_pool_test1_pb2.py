# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/protobuf/internal/descriptor_pool_test1.proto

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
  name='google/protobuf/internal/descriptor_pool_test1.proto',
  package='google.protobuf.python.internal',
  syntax='proto2',
  serialized_pb=_b('\n4google/protobuf/internal/descriptor_pool_test1.proto\x12\x1fgoogle.protobuf.python.internal\"\xfb\x05\n\x13\x44\x65scriptorPoolTest1\x12Z\n\x0bnested_enum\x18\x01 \x01(\x0e\x32?.google.protobuf.python.internal.DescriptorPoolTest1.NestedEnum:\x04\x42\x45TA\x12Z\n\x0enested_message\x18\x02 \x01(\x0b\x32\x42.google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage\x1a\xfd\x03\n\rNestedMessage\x12h\n\x0bnested_enum\x18\x01 \x01(\x0e\x32M.google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.NestedEnum:\x04ZETA\x12\x1a\n\x0cnested_field\x18\x02 \x01(\t:\x04\x62\x65ta\x12q\n\x13\x64\x65\x65p_nested_message\x18\x03 \x01(\x0b\x32T.google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage\x1a\xcd\x01\n\x11\x44\x65\x65pNestedMessage\x12y\n\x0bnested_enum\x18\x01 \x01(\x0e\x32_.google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage.NestedEnum:\x03\x45TA\x12\x1b\n\x0cnested_field\x18\x02 \x01(\t:\x05theta\" \n\nNestedEnum\x12\x07\n\x03\x45TA\x10\x07\x12\t\n\x05THETA\x10\x08\"#\n\nNestedEnum\x12\x0b\n\x07\x45PSILON\x10\x05\x12\x08\n\x04ZETA\x10\x06\"!\n\nNestedEnum\x12\t\n\x05\x41LPHA\x10\x01\x12\x08\n\x04\x42\x45TA\x10\x02*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xf1\x05\n\x13\x44\x65scriptorPoolTest2\x12[\n\x0bnested_enum\x18\x01 \x01(\x0e\x32?.google.protobuf.python.internal.DescriptorPoolTest2.NestedEnum:\x05GAMMA\x12Z\n\x0enested_message\x18\x02 \x01(\x0b\x32\x42.google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage\x1a\xfc\x03\n\rNestedMessage\x12h\n\x0bnested_enum\x18\x01 \x01(\x0e\x32M.google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.NestedEnum:\x04IOTA\x12\x1b\n\x0cnested_field\x18\x02 \x01(\t:\x05\x64\x65lta\x12q\n\x13\x64\x65\x65p_nested_message\x18\x03 \x01(\x0b\x32T.google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage\x1a\xcd\x01\n\x11\x44\x65\x65pNestedMessage\x12x\n\x0bnested_enum\x18\x01 \x01(\x0e\x32_.google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage.NestedEnum:\x02MU\x12\x1c\n\x0cnested_field\x18\x02 \x01(\t:\x06lambda\" \n\nNestedEnum\x12\n\n\x06LAMBDA\x10\x0b\x12\x06\n\x02MU\x10\x0c\"!\n\nNestedEnum\x12\x08\n\x04IOTA\x10\t\x12\t\n\x05KAPPA\x10\n\"\"\n\nNestedEnum\x12\t\n\x05GAMMA\x10\x03\x12\t\n\x05\x44\x45LTA\x10\x04')
)



_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ETA', index=0, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='THETA', index=1, number=8,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=738,
  serialized_end=770,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM)

_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='EPSILON', index=0, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ZETA', index=1, number=6,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=772,
  serialized_end=807,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_NESTEDENUM)

_DESCRIPTORPOOLTEST1_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ALPHA', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BETA', index=1, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=809,
  serialized_end=842,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST1_NESTEDENUM)

_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LAMBDA', index=0, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MU', index=1, number=12,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1506,
  serialized_end=1538,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM)

_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IOTA', index=0, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KAPPA', index=1, number=10,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1540,
  serialized_end=1573,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_NESTEDENUM)

_DESCRIPTORPOOLTEST2_NESTEDENUM = _descriptor.EnumDescriptor(
  name='NestedEnum',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GAMMA', index=0, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DELTA', index=1, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1575,
  serialized_end=1609,
)
_sym_db.RegisterEnumDescriptor(_DESCRIPTORPOOLTEST2_NESTEDENUM)


_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE = _descriptor.Descriptor(
  name='DeepNestedMessage',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=7,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_field', full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage.nested_field', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("theta").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=565,
  serialized_end=770,
)

_DESCRIPTORPOOLTEST1_NESTEDMESSAGE = _descriptor.Descriptor(
  name='NestedMessage',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=6,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_field', full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.nested_field', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("beta").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deep_nested_message', full_name='google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.deep_nested_message', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE, ],
  enum_types=[
    _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=298,
  serialized_end=807,
)

_DESCRIPTORPOOLTEST1 = _descriptor.Descriptor(
  name='DescriptorPoolTest1',
  full_name='google.protobuf.python.internal.DescriptorPoolTest1',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest1.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_message', full_name='google.protobuf.python.internal.DescriptorPoolTest1.nested_message', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DESCRIPTORPOOLTEST1_NESTEDMESSAGE, ],
  enum_types=[
    _DESCRIPTORPOOLTEST1_NESTEDENUM,
  ],
  options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 536870912), ],
  oneofs=[
  ],
  serialized_start=90,
  serialized_end=853,
)


_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE = _descriptor.Descriptor(
  name='DeepNestedMessage',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=12,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_field', full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage.nested_field', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("lambda").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1333,
  serialized_end=1538,
)

_DESCRIPTORPOOLTEST2_NESTEDMESSAGE = _descriptor.Descriptor(
  name='NestedMessage',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_field', full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.nested_field', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("delta").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deep_nested_message', full_name='google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.deep_nested_message', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE, ],
  enum_types=[
    _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1065,
  serialized_end=1573,
)

_DESCRIPTORPOOLTEST2 = _descriptor.Descriptor(
  name='DescriptorPoolTest2',
  full_name='google.protobuf.python.internal.DescriptorPoolTest2',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nested_enum', full_name='google.protobuf.python.internal.DescriptorPoolTest2.nested_enum', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nested_message', full_name='google.protobuf.python.internal.DescriptorPoolTest2.nested_message', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DESCRIPTORPOOLTEST2_NESTEDMESSAGE, ],
  enum_types=[
    _DESCRIPTORPOOLTEST2_NESTEDENUM,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=856,
  serialized_end=1609,
)

_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE.containing_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_NESTEDENUM
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE.fields_by_name['deep_nested_message'].message_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE.containing_type = _DESCRIPTORPOOLTEST1
_DESCRIPTORPOOLTEST1_NESTEDMESSAGE_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE
_DESCRIPTORPOOLTEST1.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST1_NESTEDENUM
_DESCRIPTORPOOLTEST1.fields_by_name['nested_message'].message_type = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE
_DESCRIPTORPOOLTEST1_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST1
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE.containing_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_NESTEDENUM
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE.fields_by_name['deep_nested_message'].message_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE.containing_type = _DESCRIPTORPOOLTEST2
_DESCRIPTORPOOLTEST2_NESTEDMESSAGE_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE
_DESCRIPTORPOOLTEST2.fields_by_name['nested_enum'].enum_type = _DESCRIPTORPOOLTEST2_NESTEDENUM
_DESCRIPTORPOOLTEST2.fields_by_name['nested_message'].message_type = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE
_DESCRIPTORPOOLTEST2_NESTEDENUM.containing_type = _DESCRIPTORPOOLTEST2
DESCRIPTOR.message_types_by_name['DescriptorPoolTest1'] = _DESCRIPTORPOOLTEST1
DESCRIPTOR.message_types_by_name['DescriptorPoolTest2'] = _DESCRIPTORPOOLTEST2
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DescriptorPoolTest1 = _reflection.GeneratedProtocolMessageType('DescriptorPoolTest1', (_message.Message,), dict(

  NestedMessage = _reflection.GeneratedProtocolMessageType('NestedMessage', (_message.Message,), dict(

    DeepNestedMessage = _reflection.GeneratedProtocolMessageType('DeepNestedMessage', (_message.Message,), dict(
      DESCRIPTOR = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE_DEEPNESTEDMESSAGE,
      __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
      # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage.DeepNestedMessage)
      ))
    ,
    DESCRIPTOR = _DESCRIPTORPOOLTEST1_NESTEDMESSAGE,
    __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
    # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest1.NestedMessage)
    ))
  ,
  DESCRIPTOR = _DESCRIPTORPOOLTEST1,
  __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest1)
  ))
_sym_db.RegisterMessage(DescriptorPoolTest1)
_sym_db.RegisterMessage(DescriptorPoolTest1.NestedMessage)
_sym_db.RegisterMessage(DescriptorPoolTest1.NestedMessage.DeepNestedMessage)

DescriptorPoolTest2 = _reflection.GeneratedProtocolMessageType('DescriptorPoolTest2', (_message.Message,), dict(

  NestedMessage = _reflection.GeneratedProtocolMessageType('NestedMessage', (_message.Message,), dict(

    DeepNestedMessage = _reflection.GeneratedProtocolMessageType('DeepNestedMessage', (_message.Message,), dict(
      DESCRIPTOR = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE_DEEPNESTEDMESSAGE,
      __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
      # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage.DeepNestedMessage)
      ))
    ,
    DESCRIPTOR = _DESCRIPTORPOOLTEST2_NESTEDMESSAGE,
    __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
    # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest2.NestedMessage)
    ))
  ,
  DESCRIPTOR = _DESCRIPTORPOOLTEST2,
  __module__ = 'google.protobuf.internal.descriptor_pool_test1_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.python.internal.DescriptorPoolTest2)
  ))
_sym_db.RegisterMessage(DescriptorPoolTest2)
_sym_db.RegisterMessage(DescriptorPoolTest2.NestedMessage)
_sym_db.RegisterMessage(DescriptorPoolTest2.NestedMessage.DeepNestedMessage)


# @@protoc_insertion_point(module_scope)
