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

"""Contains container classes to represent different protocol buffer types.

This file defines container classes which represent categories of protocol
buffer field types which need extra maintenance. Lwrrently these categories
are:
  - Repeated scalar fields - These are all repeated fields which aren't
    composite (e.g. they are of simple types like int32, string, etc).
  - Repeated composite fields - Repeated fields which are composite. This
    includes groups and nested messages.
"""

__author__ = 'petar@google.com (Petar Petrov)'

import collections
import sys

if sys.version_info[0] < 3:
  # We would use collections.MutableMapping all the time, but in Python 2 it
  # doesn't define __slots__.  This causes two significant problems:
  #
  # 1. we can't disallow arbitrary attribute assignment, even if our derived
  #    classes *do* define __slots__.
  #
  # 2. we can't safely derive a C type from it without __slots__ defined (the
  #    interpreter expects to find a dict at tp_dictoffset, which we can't
  #    robustly provide.  And we don't want an instance dict anyway.
  #
  # So this is the Python 2.7 definition of Mapping/MutableMapping functions
  # verbatim, except that:
  # 1. We declare __slots__.
  # 2. We don't declare this as a virtual base class.  The classes defined
  #    in collections are the interesting base classes, not us.
  #
  # Note: deriving from object is critical.  It is the only thing that makes
  # this a true type, allowing us to derive from it in C++ cleanly and making
  # __slots__ properly disallow arbitrary element assignment.

  class Mapping(object):
    __slots__ = ()

    def get(self, key, default=None):
      try:
        return self[key]
      except KeyError:
        return default

    def __contains__(self, key):
      try:
        self[key]
      except KeyError:
        return False
      else:
        return True

    def iterkeys(self):
      return iter(self)

    def itervalues(self):
      for key in self:
        yield self[key]

    def iteritems(self):
      for key in self:
        yield (key, self[key])

    def keys(self):
      return list(self)

    def items(self):
      return [(key, self[key]) for key in self]

    def values(self):
      return [self[key] for key in self]

    # Mappings are not hashable by default, but subclasses can change this
    __hash__ = None

    def __eq__(self, other):
      if not isinstance(other, collections.Mapping):
        return NotImplemented
      return dict(self.items()) == dict(other.items())

    def __ne__(self, other):
      return not (self == other)

  class MutableMapping(Mapping):
    __slots__ = ()

    __marker = object()

    def pop(self, key, default=__marker):
      try:
        value = self[key]
      except KeyError:
        if default is self.__marker:
          raise
        return default
      else:
        del self[key]
        return value

    def popitem(self):
      try:
        key = next(iter(self))
      except StopIteration:
        raise KeyError
      value = self[key]
      del self[key]
      return key, value

    def clear(self):
      try:
        while True:
          self.popitem()
      except KeyError:
        pass

    def update(*args, **kwds):
      if len(args) > 2:
        raise TypeError("update() takes at most 2 positional "
                        "arguments ({} given)".format(len(args)))
      elif not args:
        raise TypeError("update() takes at least 1 argument (0 given)")
      self = args[0]
      other = args[1] if len(args) >= 2 else ()

      if isinstance(other, Mapping):
        for key in other:
          self[key] = other[key]
      elif hasattr(other, "keys"):
        for key in other.keys():
          self[key] = other[key]
      else:
        for key, value in other:
          self[key] = value
      for key, value in kwds.items():
        self[key] = value

    def setdefault(self, key, default=None):
      try:
        return self[key]
      except KeyError:
        self[key] = default
      return default

  collections.Mapping.register(Mapping)
  collections.MutableMapping.register(MutableMapping)

else:
  # In Python 3 we can just use MutableMapping directly, because it defines
  # __slots__.
  MutableMapping = collections.MutableMapping


class BaseContainer(object):

  """Base container class."""

  # Minimizes memory usage and disallows assignment to other attributes.
  __slots__ = ['_message_listener', '_values']

  def __init__(self, message_listener):
    """
    Args:
      message_listener: A MessageListener implementation.
        The RepeatedScalarFieldContainer will call this object's
        Modified() method when it is modified.
    """
    self._message_listener = message_listener
    self._values = []

  def __getitem__(self, key):
    """Retrieves item by the specified key."""
    return self._values[key]

  def __len__(self):
    """Returns the number of elements in the container."""
    return len(self._values)

  def __ne__(self, other):
    """Checks if another instance isn't equal to this one."""
    # The concrete classes should define __eq__.
    return not self == other

  def __hash__(self):
    raise TypeError('unhashable object')

  def __repr__(self):
    return repr(self._values)

  def sort(self, *args, **kwargs):
    # Continue to support the old sort_function keyword argument.
    # This is expected to be a rare oclwrrence, so use LBYL to avoid
    # the overhead of actually catching KeyError.
    if 'sort_function' in kwargs:
      kwargs['cmp'] = kwargs.pop('sort_function')
    self._values.sort(*args, **kwargs)


class RepeatedScalarFieldContainer(BaseContainer):

  """Simple, type-checked, list-like container for holding repeated scalars."""

  # Disallows assignment to other attributes.
  __slots__ = ['_type_checker']

  def __init__(self, message_listener, type_checker):
    """
    Args:
      message_listener: A MessageListener implementation.
        The RepeatedScalarFieldContainer will call this object's
        Modified() method when it is modified.
      type_checker: A type_checkers.ValueChecker instance to run on elements
        inserted into this container.
    """
    super(RepeatedScalarFieldContainer, self).__init__(message_listener)
    self._type_checker = type_checker

  def append(self, value):
    """Appends an item to the list. Similar to list.append()."""
    self._values.append(self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def insert(self, key, value):
    """Inserts the item at the specified position. Similar to list.insert()."""
    self._values.insert(key, self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def extend(self, elem_seq):
    """Extends by appending the given iterable. Similar to list.extend()."""

    if elem_seq is None:
      return
    try:
      elem_seq_iter = iter(elem_seq)
    except TypeError:
      if not elem_seq:
        # silently ignore falsy inputs :-/.
        # TODO(ptucker): Deprecate this behavior. b/18413862
        return
      raise

    new_values = [self._type_checker.CheckValue(elem) for elem in elem_seq_iter]
    if new_values:
      self._values.extend(new_values)
    self._message_listener.Modified()

  def MergeFrom(self, other):
    """Appends the contents of another repeated field of the same type to this
    one. We do not check the types of the individual fields.
    """
    self._values.extend(other._values)
    self._message_listener.Modified()

  def remove(self, elem):
    """Removes an item from the list. Similar to list.remove()."""
    self._values.remove(elem)
    self._message_listener.Modified()

  def pop(self, key=-1):
    """Removes and returns an item at a given index. Similar to list.pop()."""
    value = self._values[key]
    self.__delitem__(key)
    return value

  def __setitem__(self, key, value):
    """Sets the item on the specified position."""
    if isinstance(key, slice):  # PY3
      if key.step is not None:
        raise ValueError('Extended slices not supported')
      self.__setslice__(key.start, key.stop, value)
    else:
      self._values[key] = self._type_checker.CheckValue(value)
      self._message_listener.Modified()

  def __getslice__(self, start, stop):
    """Retrieves the subset of items from between the specified indices."""
    return self._values[start:stop]

  def __setslice__(self, start, stop, values):
    """Sets the subset of items from between the specified indices."""
    new_values = []
    for value in values:
      new_values.append(self._type_checker.CheckValue(value))
    self._values[start:stop] = new_values
    self._message_listener.Modified()

  def __delitem__(self, key):
    """Deletes the item at the specified position."""
    del self._values[key]
    self._message_listener.Modified()

  def __delslice__(self, start, stop):
    """Deletes the subset of items from between the specified indices."""
    del self._values[start:stop]
    self._message_listener.Modified()

  def __eq__(self, other):
    """Compares the current instance with another one."""
    if self is other:
      return True
    # Special case for the same type which should be common and fast.
    if isinstance(other, self.__class__):
      return other._values == self._values
    # We are presumably comparing against some other sequence type.
    return other == self._values

collections.MutableSequence.register(BaseContainer)


class RepeatedCompositeFieldContainer(BaseContainer):

  """Simple, list-like container for holding repeated composite fields."""

  # Disallows assignment to other attributes.
  __slots__ = ['_message_descriptor']

  def __init__(self, message_listener, message_descriptor):
    """
    Note that we pass in a descriptor instead of the generated directly,
    since at the time we construct a _RepeatedCompositeFieldContainer we
    haven't yet necessarily initialized the type that will be contained in the
    container.

    Args:
      message_listener: A MessageListener implementation.
        The RepeatedCompositeFieldContainer will call this object's
        Modified() method when it is modified.
      message_descriptor: A Descriptor instance describing the protocol type
        that should be present in this container.  We'll use the
        _concrete_class field of this descriptor when the client calls add().
    """
    super(RepeatedCompositeFieldContainer, self).__init__(message_listener)
    self._message_descriptor = message_descriptor

  def add(self, **kwargs):
    """Adds a new element at the end of the list and returns it. Keyword
    arguments may be used to initialize the element.
    """
    new_element = self._message_descriptor._concrete_class(**kwargs)
    new_element._SetListener(self._message_listener)
    self._values.append(new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()
    return new_element

  def extend(self, elem_seq):
    """Extends by appending the given sequence of elements of the same type
    as this one, copying each individual message.
    """
    message_class = self._message_descriptor._concrete_class
    listener = self._message_listener
    values = self._values
    for message in elem_seq:
      new_element = message_class()
      new_element._SetListener(listener)
      new_element.MergeFrom(message)
      values.append(new_element)
    listener.Modified()

  def MergeFrom(self, other):
    """Appends the contents of another repeated field of the same type to this
    one, copying each individual message.
    """
    self.extend(other._values)

  def remove(self, elem):
    """Removes an item from the list. Similar to list.remove()."""
    self._values.remove(elem)
    self._message_listener.Modified()

  def pop(self, key=-1):
    """Removes and returns an item at a given index. Similar to list.pop()."""
    value = self._values[key]
    self.__delitem__(key)
    return value

  def __getslice__(self, start, stop):
    """Retrieves the subset of items from between the specified indices."""
    return self._values[start:stop]

  def __delitem__(self, key):
    """Deletes the item at the specified position."""
    del self._values[key]
    self._message_listener.Modified()

  def __delslice__(self, start, stop):
    """Deletes the subset of items from between the specified indices."""
    del self._values[start:stop]
    self._message_listener.Modified()

  def __eq__(self, other):
    """Compares the current instance with another one."""
    if self is other:
      return True
    if not isinstance(other, self.__class__):
      raise TypeError('Can only compare repeated composite fields against '
                      'other repeated composite fields.')
    return self._values == other._values


class ScalarMap(MutableMapping):

  """Simple, type-checked, dict-like container for holding repeated scalars."""

  # Disallows assignment to other attributes.
  __slots__ = ['_key_checker', '_value_checker', '_values', '_message_listener',
               '_entry_descriptor']

  def __init__(self, message_listener, key_checker, value_checker,
               entry_descriptor):
    """
    Args:
      message_listener: A MessageListener implementation.
        The ScalarMap will call this object's Modified() method when it
        is modified.
      key_checker: A type_checkers.ValueChecker instance to run on keys
        inserted into this container.
      value_checker: A type_checkers.ValueChecker instance to run on values
        inserted into this container.
      entry_descriptor: The MessageDescriptor of a map entry: key and value.
    """
    self._message_listener = message_listener
    self._key_checker = key_checker
    self._value_checker = value_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}

  def __getitem__(self, key):
    try:
      return self._values[key]
    except KeyError:
      key = self._key_checker.CheckValue(key)
      val = self._value_checker.DefaultValue()
      self._values[key] = val
      return val

  def __contains__(self, item):
    # We check the key's type to match the strong-typing flavor of the API.
    # Also this makes it easier to match the behavior of the C++ implementation.
    self._key_checker.CheckValue(item)
    return item in self._values

  # We need to override this explicitly, because our defaultdict-like behavior
  # will make the default implementation (from our base class) always insert
  # the key.
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default

  def __setitem__(self, key, value):
    checked_key = self._key_checker.CheckValue(key)
    checked_value = self._value_checker.CheckValue(value)
    self._values[checked_key] = checked_value
    self._message_listener.Modified()

  def __delitem__(self, key):
    del self._values[key]
    self._message_listener.Modified()

  def __len__(self):
    return len(self._values)

  def __iter__(self):
    return iter(self._values)

  def __repr__(self):
    return repr(self._values)

  def MergeFrom(self, other):
    self._values.update(other._values)
    self._message_listener.Modified()

  def IlwalidateIterators(self):
    # It appears that the only way to reliably ilwalidate iterators to
    # self._values is to ensure that its size changes.
    original = self._values
    self._values = original.copy()
    original[None] = None

  # This is defined in the abstract base, but we can do it much more cheaply.
  def clear(self):
    self._values.clear()
    self._message_listener.Modified()

  def GetEntryClass(self):
    return self._entry_descriptor._concrete_class


class MessageMap(MutableMapping):

  """Simple, type-checked, dict-like container for with submessage values."""

  # Disallows assignment to other attributes.
  __slots__ = ['_key_checker', '_values', '_message_listener',
               '_message_descriptor', '_entry_descriptor']

  def __init__(self, message_listener, message_descriptor, key_checker,
               entry_descriptor):
    """
    Args:
      message_listener: A MessageListener implementation.
        The ScalarMap will call this object's Modified() method when it
        is modified.
      key_checker: A type_checkers.ValueChecker instance to run on keys
        inserted into this container.
      value_checker: A type_checkers.ValueChecker instance to run on values
        inserted into this container.
      entry_descriptor: The MessageDescriptor of a map entry: key and value.
    """
    self._message_listener = message_listener
    self._message_descriptor = message_descriptor
    self._key_checker = key_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}

  def __getitem__(self, key):
    try:
      return self._values[key]
    except KeyError:
      key = self._key_checker.CheckValue(key)
      new_element = self._message_descriptor._concrete_class()
      new_element._SetListener(self._message_listener)
      self._values[key] = new_element
      self._message_listener.Modified()

      return new_element

  def get_or_create(self, key):
    """get_or_create() is an alias for getitem (ie. map[key]).

    Args:
      key: The key to get or create in the map.

    This is useful in cases where you want to be explicit that the call is
    mutating the map.  This can avoid lint errors for statements like this
    that otherwise would appear to be pointless statements:

      msg.my_map[key]
    """
    return self[key]

  # We need to override this explicitly, because our defaultdict-like behavior
  # will make the default implementation (from our base class) always insert
  # the key.
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default

  def __contains__(self, item):
    return item in self._values

  def __setitem__(self, key, value):
    raise ValueError('May not set values directly, call my_map[key].foo = 5')

  def __delitem__(self, key):
    del self._values[key]
    self._message_listener.Modified()

  def __len__(self):
    return len(self._values)

  def __iter__(self):
    return iter(self._values)

  def __repr__(self):
    return repr(self._values)

  def MergeFrom(self, other):
    for key in other:
      # According to documentation: "When parsing from the wire or when merging,
      # if there are duplicate map keys the last key seen is used".
      if key in self:
        del self[key]
      self[key].CopyFrom(other[key])
    # self._message_listener.Modified() not required here, because
    # mutations to submessages already propagate.

  def IlwalidateIterators(self):
    # It appears that the only way to reliably ilwalidate iterators to
    # self._values is to ensure that its size changes.
    original = self._values
    self._values = original.copy()
    original[None] = None

  # This is defined in the abstract base, but we can do it much more cheaply.
  def clear(self):
    self._values.clear()
    self._message_listener.Modified()

  def GetEntryClass(self):
    return self._entry_descriptor._concrete_class
