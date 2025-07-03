#!/usr/bin/elw python3

# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2020 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

import sys, re

proto_re = re.compile(r'[a-zA-Z_]\w*|\d+|[=;{}]|"[^"]*"')

def main():
    opened = False

    # Handle mle.proto either from command line or from stdin
    if len(sys.argv) > 1:
        if len(sys.argv) > 2 or sys.argv[1] in ("-h", "-help", "--help"):
            print("Colwerts mle.proto file to C++ header")
            print()
            print("Usage: mle.py [mle.proto]")
            print()
            print("If no arguments are specified, reads mle.proto from stdin!")
            sys.exit(0)

        f = open(sys.argv[1], "r")
        opened = True
    else:
        f = sys.stdin

    # Parse the mle.proto file - the result of parsing is Abstract Syntax Tree
    lexer = Lexer(f)
    root = ASTNode(ASTNode.root, "", None)
    Parse(lexer, root)

    if opened:
        f.close()

    # Rearrange messages (structures) so that they are usable in C/C++
    root.MarkUse()
    root.Rearrange()

    # Generate C/C++ macros
    EmitCode(root)

def Tokenize(f):
    """Generator, which yields subsequent tokens in the form of: (line_no, token_str)"""
    line_no = 0
    for line in f.readlines():
        line_no += 1

        # Remove comments and leading/trailing spaces
        line = line[:line.find("//")].strip()

        # Skip empty lines
        if len(line) == 0:
            continue

        for token in proto_re.findall(line):
            yield (line_no, token)

class Lexer:
    """Lexer (a.k.a. scanner).  Used for peeking and retrieving subsequent tokens."""
    def __init__(self, f):
        self._tokens  = Tokenize(f)
        self._next    = None
        self._line_no = 0

    def Peek(self):
        if self._next == None:
            try:
                line_no, token = next(self._tokens)
                self._next     = token
                self._line_no  = line_no
            except StopIteration:
                self._next = ""
        return self._next

    def Next(self):
        token = self.Peek()
        self._next = None
        return token

    def Line(self):
        return self._line_no

class ASTNode:
    """Node of an Abtract Syntax Tree representing the parsed mle.proto file."""
    # Type of an AST node
    root    = 0
    message = 1
    enum    = 2
    # These are the only primitive protobuf types we recognize/support
    primitive_types = ("bool", "float", "uint32", "uint64", "sint32", "sint64", "string", "bytes")

    def __init__(self, type, name, parent_node):
        self._type    = type
        self._name    = name
        self._parent  = parent_node
        if type == self.root:
            self._structs = []
            self._enums   = []
        elif type == self.message:
            self._fields         = []
            self._structs        = []
            self._enums          = []
            self._siblings_using = []
            parent_node._structs.append(self)
        else:
            self._values = []
            parent_node._enums.append(self)

    def AddEnumValue(self, key, value):
        self._values.append((key, value))

    def AddField(self, line_no, repeated, type, name, index):
        self._fields.append((line_no, repeated, type, name, index))

    def MarkUse(self):
        if self._type == self.message:
            for f in self._fields:
                self.MarkTypeUse(f[0], f[2])
        for s in self._structs:
            s.MarkUse()

    def MarkTypeUse(self, line_no, type):
        """For each structure, find out which other sibling structures are using it."""
        if type in self.primitive_types:
            return

        found = None
        node  = self
        prev  = None
        while node != None:
            for s in node._structs:
                if s._name == type:
                    found = s
                    break
            if not found:
                for e in node._enums:
                    if e._name == type:
                        found = e
                        break
            if found:
                break
            prev = node._name
            node = node._parent
        if found:
            if prev and found._type == self.message:
                found._siblings_using.append(prev)
        else:
            print("Line {}: unrecognized type '{}'".format(line_no, type), file=sys.stderr)
            sys.exit(1)

    def Rearrange(self):
        """Rearrange structures, so that sibling structures using it are declared after it."""
        for s in self._structs:
            s.Rearrange()

        total = len(self._structs)

        while True:
            src  = None
            dest = None

            for i in range(total):
                name = self._structs[i]._name
                largest_ref = i
                for j in range(i + 1, total):
                    if name in self._structs[j]._siblings_using:
                        largest_ref = j
                if largest_ref > i:
                    src  = i
                    dest = largest_ref
                    break

            if dest:
                self._structs.insert(dest, self._structs.pop(src))
            else:
                break

def Expect(lexer, *values):
    """Expect the lexer to yield specific tokens in the specified sequence."""
    ret = ()
    for v in values:
        token = lexer.Next()
        if v == None:
            ret += (token,)
        elif token != v:
            print("Line {}: '{}' expected, but found '{}'".format(lexer.Line(), v, token), file=sys.stderr)
            sys.exit(1)
    return ret

def Parse(lexer, node):
    """Parse a proto file and produce Abstract Syntax Tree."""
    Expect(lexer, "syntax", "=", '"proto3"', ";")

    while True:
        token = lexer.Peek()
        if token == "enum":
            ParseEnum(lexer, node)
        elif token == "":
            break
        else:
            ParseMessage(lexer, node)

def ParseMessage(lexer, node):
    name, = Expect(lexer, "message", None)
    new_node = ASTNode(ASTNode.message, name, node)
    Expect(lexer, "{")
    while True:
        token = lexer.Peek()
        if token == "message":
            ParseMessage(lexer, new_node)
        elif token == "enum":
            ParseEnum(lexer, new_node)
        elif token == "reserved":
            ParseReserved(lexer)
        elif token in ("", "}"):
            break
        else:
            ParseField(lexer, new_node)
    Expect(lexer, "}")

def ParseEnum(lexer, node):
    name, = Expect(lexer, "enum", None)
    new_node = ASTNode(ASTNode.enum, name, node)
    Expect(lexer, "{")
    while True:
        token = lexer.Peek()
        if token in ("", "}"):
            break
        key, value = Expect(lexer, None, "=", None, ";")
        new_node.AddEnumValue(key, value)
    Expect(lexer, "}")

def ParseReserved(lexer):
    min, = Expect(lexer, "reserved", None)
    max = min
    if lexer.Peek() == "to":
        lexer.Next()
        max = lexer.Next()
    Expect(lexer, ";")

def ParseField(lexer, node):
    type     = lexer.Next()
    line_no  = lexer.Line()
    repeated = type == "repeated"
    if repeated:
        type = lexer.Next()
    name, index = Expect(lexer, None, "=", None, ";")
    node.AddField(line_no, repeated, type, name, index)

def EmitCode(node, indent_level = -1):
    """Produce macro ilwocations for use in C/C++ sources."""
    indent_base = " " * 4
    indent = indent_base * indent_level
    if node._type == ASTNode.enum:
        print("{}BEGIN_ENUM({})".format(indent, node._name))
        for k, v in node._values:
            print("{}{}DEFINE_ENUM_VALUE({}, {})".format(indent, indent_base, k, v))
        print("{}END_ENUM".format(indent))
    else:
        if node._type == ASTNode.message:
            print("{}BEGIN_MESSAGE({})".format(indent, node._name))
        for e in node._enums:
            EmitCode(e, indent_level + 1)
        for s in node._structs:
            EmitCode(s, indent_level + 1)
        if node._type == ASTNode.message:
            for line_no, repeated, f_type, f_name, f_index in node._fields:
                if f_type in ASTNode.primitive_types:
                    f_type = "pb_" + f_type
                print("{}{}DEFINE_{}FIELD({}, {}, {})".format(
                      indent, indent_base, "REPEATED_" if repeated else "", f_name, f_type, f_index))
            print("{}END_MESSAGE".format(indent))

if __name__ == "__main__":
    main()
