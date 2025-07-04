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

# TODO(robinson): We should just make these methods all "pure-virtual" and move
# all implementation out, into reflection.py for now.


"""Contains an abstract base class for protocol messages."""

__author__ = 'robinson@google.com (Will Robinson)'

class Error(Exception): pass
class DecodeError(Error): pass
class EncodeError(Error): pass


class Message(object):

  """Abstract base class for protocol messages.

  Protocol message classes are almost always generated by the protocol
  compiler.  These generated types subclass Message and implement the methods
  shown below.

  TODO(robinson): Link to an HTML document here.

  TODO(robinson): Document that instances of this class will also
  have an Extensions attribute with __getitem__ and __setitem__.
  Again, not sure how to best convey this.

  TODO(robinson): Document that the class must also have a static
    RegisterExtension(extension_field) method.
    Not sure how to best express at this point.
  """

  # TODO(robinson): Document these fields and methods.

  __slots__ = []

  DESCRIPTOR = None

  def __deepcopy__(self, memo=None):
    clone = type(self)()
    clone.MergeFrom(self)
    return clone

  def __eq__(self, other_msg):
    """Relwrsively compares two messages by value and structure."""
    raise NotImplementedError

  def __ne__(self, other_msg):
    # Can't just say self != other_msg, since that would infinitely relwrse. :)
    return not self == other_msg

  def __hash__(self):
    raise TypeError('unhashable object')

  def __str__(self):
    """Outputs a human-readable representation of the message."""
    raise NotImplementedError

  def __unicode__(self):
    """Outputs a human-readable representation of the message."""
    raise NotImplementedError

  def MergeFrom(self, other_msg):
    """Merges the contents of the specified message into current message.

    This method merges the contents of the specified message into the current
    message. Singular fields that are set in the specified message overwrite
    the corresponding fields in the current message. Repeated fields are
    appended. Singular sub-messages and groups are relwrsively merged.

    Args:
      other_msg: Message to merge into the current message.
    """
    raise NotImplementedError

  def CopyFrom(self, other_msg):
    """Copies the content of the specified message into the current message.

    The method clears the current message and then merges the specified
    message using MergeFrom.

    Args:
      other_msg: Message to copy into the current one.
    """
    if self is other_msg:
      return
    self.Clear()
    self.MergeFrom(other_msg)

  def Clear(self):
    """Clears all data that was set in the message."""
    raise NotImplementedError

  def SetInParent(self):
    """Mark this as present in the parent.

    This normally happens automatically when you assign a field of a
    sub-message, but sometimes you want to make the sub-message
    present while keeping it empty.  If you find yourself using this,
    you may want to reconsider your design."""
    raise NotImplementedError

  def IsInitialized(self):
    """Checks if the message is initialized.

    Returns:
      The method returns True if the message is initialized (i.e. all of its
      required fields are set).
    """
    raise NotImplementedError

  # TODO(robinson): MergeFromString() should probably return None and be
  # implemented in terms of a helper that returns the # of bytes read.  Our
  # deserialization routines would use the helper when relwrsively
  # deserializing, but the end user would almost always just want the no-return
  # MergeFromString().

  def MergeFromString(self, serialized):
    """Merges serialized protocol buffer data into this message.

    When we find a field in |serialized| that is already present
    in this message:
      - If it's a "repeated" field, we append to the end of our list.
      - Else, if it's a scalar, we overwrite our field.
      - Else, (it's a nonrepeated composite), we relwrsively merge
        into the existing composite.

    TODO(robinson): Document handling of unknown fields.

    Args:
      serialized: Any object that allows us to call buffer(serialized)
        to access a string of bytes using the buffer interface.

    TODO(robinson): When we switch to a helper, this will return None.

    Returns:
      The number of bytes read from |serialized|.
      For non-group messages, this will always be len(serialized),
      but for messages which are actually groups, this will
      generally be less than len(serialized), since we must
      stop when we reach an END_GROUP tag.  Note that if
      we *do* stop because of an END_GROUP tag, the number
      of bytes returned does not include the bytes
      for the END_GROUP tag information.
    """
    raise NotImplementedError

  def ParseFromString(self, serialized):
    """Parse serialized protocol buffer data into this message.

    Like MergeFromString(), except we clear the object first and
    do not return the value that MergeFromString returns.
    """
    self.Clear()
    self.MergeFromString(serialized)

  def SerializeToString(self, **kwargs):
    """Serializes the protocol message to a binary string.

    Arguments:
      **kwargs: Keyword arguments to the serialize method, accepts
        the following keyword args:
        deterministic: If true, requests deterministic serialization of the
          protobuf, with predictable ordering of map keys.

    Returns:
      A binary string representation of the message if all of the required
      fields in the message are set (i.e. the message is initialized).

    Raises:
      message.EncodeError if the message isn't initialized.
    """
    raise NotImplementedError

  def SerializePartialToString(self, **kwargs):
    """Serializes the protocol message to a binary string.

    This method is similar to SerializeToString but doesn't check if the
    message is initialized.

    Arguments:
      **kwargs: Keyword arguments to the serialize method, accepts
        the following keyword args:
        deterministic: If true, requests deterministic serialization of the
          protobuf, with predictable ordering of map keys.

    Returns:
      A string representation of the partial message.
    """
    raise NotImplementedError

  # TODO(robinson): Decide whether we like these better
  # than auto-generated has_foo() and clear_foo() methods
  # on the instances themselves.  This way is less consistent
  # with C++, but it makes reflection-type access easier and
  # reduces the number of magically autogenerated things.
  #
  # TODO(robinson): Be sure to document (and test) exactly
  # which field names are accepted here.  Are we case-sensitive?
  # What do we do with fields that share names with Python keywords
  # like 'lambda' and 'yield'?
  #
  # nnorwitz says:
  # """
  # Typically (in python), an underscore is appended to names that are
  # keywords. So they would become lambda_ or yield_.
  # """
  def ListFields(self):
    """Returns a list of (FieldDescriptor, value) tuples for all
    fields in the message which are not empty.  A message field is
    non-empty if HasField() would return true. A singular primitive field
    is non-empty if HasField() would return true in proto2 or it is non zero
    in proto3. A repeated field is non-empty if it contains at least one
    element.  The fields are ordered by field number"""
    raise NotImplementedError

  def HasField(self, field_name):
    """Checks if a certain field is set for the message, or if any field inside
    a oneof group is set.  Note that if the field_name is not defined in the
    message descriptor, ValueError will be raised."""
    raise NotImplementedError

  def ClearField(self, field_name):
    """Clears the contents of a given field, or the field set inside a oneof
    group.  If the name neither refers to a defined field or oneof group,
    ValueError is raised."""
    raise NotImplementedError

  def WhichOneof(self, oneof_group):
    """Returns the name of the field that is set inside a oneof group, or
    None if no field is set.  If no group with the given name exists, ValueError
    will be raised."""
    raise NotImplementedError

  def HasExtension(self, extension_handle):
    raise NotImplementedError

  def ClearExtension(self, extension_handle):
    raise NotImplementedError

  def DiscardUnknownFields(self):
    raise NotImplementedError

  def ByteSize(self):
    """Returns the serialized size of this message.
    Relwrsively calls ByteSize() on all contained messages.
    """
    raise NotImplementedError

  def _SetListener(self, message_listener):
    """Internal method used by the protocol message implementation.
    Clients should not call this directly.

    Sets a listener that this message will call on certain state transitions.

    The purpose of this method is to register back-edges from children to
    parents at runtime, for the purpose of setting "has" bits and
    byte-size-dirty bits in the parent and ancestor objects whenever a child or
    descendant object is modified.

    If the client wants to disconnect this Message from the object tree, she
    explicitly sets callback to None.

    If message_listener is None, unregisters any existing listener.  Otherwise,
    message_listener must implement the MessageListener interface in
    internal/message_listener.py, and we discard any listener registered
    via a previous _SetListener() call.
    """
    raise NotImplementedError

  def __getstate__(self):
    """Support the pickle protocol."""
    return dict(serialized=self.SerializePartialToString())

  def __setstate__(self, state):
    """Support the pickle protocol."""
    self.__init__()
    self.ParseFromString(state['serialized'])
