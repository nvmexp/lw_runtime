# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/protobuf/internal/message_set_extensions.proto

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
  name='google/protobuf/internal/message_set_extensions.proto',
  package='google.protobuf.internal',
  syntax='proto2',
  serialized_pb=_b('\n5google/protobuf/internal/message_set_extensions.proto\x12\x18google.protobuf.internal\"\x1e\n\x0eTestMessageSet*\x08\x08\x04\x10\xff\xff\xff\xff\x07:\x02\x08\x01\"\xa5\x01\n\x18TestMessageSetExtension1\x12\t\n\x01i\x18\x0f \x01(\x05\x32~\n\x15message_set_extension\x12(.google.protobuf.internal.TestMessageSet\x18\xab\xff\xf6. \x01(\x0b\x32\x32.google.protobuf.internal.TestMessageSetExtension1\"\xa7\x01\n\x18TestMessageSetExtension2\x12\x0b\n\x03str\x18\x19 \x01(\t2~\n\x15message_set_extension\x12(.google.protobuf.internal.TestMessageSet\x18\xca\xff\xf6. \x01(\x0b\x32\x32.google.protobuf.internal.TestMessageSetExtension2\"(\n\x18TestMessageSetExtension3\x12\x0c\n\x04text\x18# \x01(\t:\x7f\n\x16message_set_extension3\x12(.google.protobuf.internal.TestMessageSet\x18\xdf\xff\xf6. \x01(\x0b\x32\x32.google.protobuf.internal.TestMessageSetExtension3')
)


MESSAGE_SET_EXTENSION3_FIELD_NUMBER = 98418655
message_set_extension3 = _descriptor.FieldDescriptor(
  name='message_set_extension3', full_name='google.protobuf.internal.message_set_extension3', index=0,
  number=98418655, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None, file=DESCRIPTOR)


_TESTMESSAGESET = _descriptor.Descriptor(
  name='TestMessageSet',
  full_name='google.protobuf.internal.TestMessageSet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('\010\001')),
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(4, 2147483647), ],
  oneofs=[
  ],
  serialized_start=83,
  serialized_end=113,
)


_TESTMESSAGESETEXTENSION1 = _descriptor.Descriptor(
  name='TestMessageSetExtension1',
  full_name='google.protobuf.internal.TestMessageSetExtension1',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='i', full_name='google.protobuf.internal.TestMessageSetExtension1.i', index=0,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='message_set_extension', full_name='google.protobuf.internal.TestMessageSetExtension1.message_set_extension', index=0,
      number=98418603, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      options=None, file=DESCRIPTOR),
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
  serialized_start=116,
  serialized_end=281,
)


_TESTMESSAGESETEXTENSION2 = _descriptor.Descriptor(
  name='TestMessageSetExtension2',
  full_name='google.protobuf.internal.TestMessageSetExtension2',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='str', full_name='google.protobuf.internal.TestMessageSetExtension2.str', index=0,
      number=25, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='message_set_extension', full_name='google.protobuf.internal.TestMessageSetExtension2.message_set_extension', index=0,
      number=98418634, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      options=None, file=DESCRIPTOR),
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
  serialized_start=284,
  serialized_end=451,
)


_TESTMESSAGESETEXTENSION3 = _descriptor.Descriptor(
  name='TestMessageSetExtension3',
  full_name='google.protobuf.internal.TestMessageSetExtension3',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='google.protobuf.internal.TestMessageSetExtension3.text', index=0,
      number=35, type=9, cpp_type=9, label=1,
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
  serialized_start=453,
  serialized_end=493,
)

DESCRIPTOR.message_types_by_name['TestMessageSet'] = _TESTMESSAGESET
DESCRIPTOR.message_types_by_name['TestMessageSetExtension1'] = _TESTMESSAGESETEXTENSION1
DESCRIPTOR.message_types_by_name['TestMessageSetExtension2'] = _TESTMESSAGESETEXTENSION2
DESCRIPTOR.message_types_by_name['TestMessageSetExtension3'] = _TESTMESSAGESETEXTENSION3
DESCRIPTOR.extensions_by_name['message_set_extension3'] = message_set_extension3
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TestMessageSet = _reflection.GeneratedProtocolMessageType('TestMessageSet', (_message.Message,), dict(
  DESCRIPTOR = _TESTMESSAGESET,
  __module__ = 'google.protobuf.internal.message_set_extensions_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.internal.TestMessageSet)
  ))
_sym_db.RegisterMessage(TestMessageSet)

TestMessageSetExtension1 = _reflection.GeneratedProtocolMessageType('TestMessageSetExtension1', (_message.Message,), dict(
  DESCRIPTOR = _TESTMESSAGESETEXTENSION1,
  __module__ = 'google.protobuf.internal.message_set_extensions_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.internal.TestMessageSetExtension1)
  ))
_sym_db.RegisterMessage(TestMessageSetExtension1)

TestMessageSetExtension2 = _reflection.GeneratedProtocolMessageType('TestMessageSetExtension2', (_message.Message,), dict(
  DESCRIPTOR = _TESTMESSAGESETEXTENSION2,
  __module__ = 'google.protobuf.internal.message_set_extensions_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.internal.TestMessageSetExtension2)
  ))
_sym_db.RegisterMessage(TestMessageSetExtension2)

TestMessageSetExtension3 = _reflection.GeneratedProtocolMessageType('TestMessageSetExtension3', (_message.Message,), dict(
  DESCRIPTOR = _TESTMESSAGESETEXTENSION3,
  __module__ = 'google.protobuf.internal.message_set_extensions_pb2'
  # @@protoc_insertion_point(class_scope:google.protobuf.internal.TestMessageSetExtension3)
  ))
_sym_db.RegisterMessage(TestMessageSetExtension3)

message_set_extension3.message_type = _TESTMESSAGESETEXTENSION3
TestMessageSet.RegisterExtension(message_set_extension3)
_TESTMESSAGESETEXTENSION1.extensions_by_name['message_set_extension'].message_type = _TESTMESSAGESETEXTENSION1
TestMessageSet.RegisterExtension(_TESTMESSAGESETEXTENSION1.extensions_by_name['message_set_extension'])
_TESTMESSAGESETEXTENSION2.extensions_by_name['message_set_extension'].message_type = _TESTMESSAGESETEXTENSION2
TestMessageSet.RegisterExtension(_TESTMESSAGESETEXTENSION2.extensions_by_name['message_set_extension'])

_TESTMESSAGESET.has_options = True
_TESTMESSAGESET._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('\010\001'))
# @@protoc_insertion_point(module_scope)
