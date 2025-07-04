#  Copyright 2008-2014 Nokia Solutions and Networks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import with_statement

import copy
import re

try:
    from lxml import etree as lxml_etree
except ImportError:
    lxml_etree = None

from robot.api import logger
from robot.libraries.BuiltIn import BuiltIn
from robot.utils import asserts, ET, ETSource, plural_or_not as s
from robot.version import get_version


should_be_equal = asserts.assert_equals
should_match = BuiltIn().should_match


class XML(object):
    """Robot Framework test library for verifying and modifying XML dolwments.

    As the name implies, _XML_ is a test library for verifying contents of XML
    files. In practice it is a pretty thin wrapper on top of Python's
    [https://docs.python.org/2/library/xml.etree.elementtree.html|ElementTree XML API].

    The library has the following main usages:

    - Parsing an XML file, or a string containing XML, into an XML element
      structure and finding certain elements from it for for further analysis
      (e.g. `Parse XML` and `Get Element` keywords).
    - Getting text or attributes of elements
      (e.g. `Get Element Text` and `Get Element Attribute`).
    - Directly verifying text, attributes, or whole elements
      (e.g `Element Text Should Be` and `Elements Should Be Equal`).
    - Modifying XML and saving it (e.g. `Set Element Text`, `Add Element`
      and `Save XML`).

    By default this library uses ElementTree module for parsing XML, but it
    can be configured to use [http://lxml.de|lxml] instead when `importing`
    the library. The main benefit of using lxml is that it supports richer
    xpath syntax than the standard ElementTree. It also enables using
    `Evaluate Xpath` keyword and preserves possible namespace prefixes when
    saving XML. The lxml support is new in Robot Framework 2.8.5.

    == Table of contents ==

    - `Parsing XML`
    - `Example`
    - `Finding elements with xpath`
    - `Element attributes`
    - `Handling XML namespaces`
    - `Shortlwts`
    - `Keywords`

    = Parsing XML =

    XML can be parsed into an element structure using `Parse XML` keyword.
    It accepts both paths to XML files and strings that contain XML. The
    keyword returns the root element of the structure, which then contains
    other elements as its children and their children.

    The element structure returned by `Parse XML`, as well as elements
    returned by keywords such as `Get Element`, can be used as the ``source``
    argument with other keywords. In addition to an already parsed XML
    structure, other keywords also accept paths to XML files and strings
    containing XML similarly as `Parse XML`. Notice that keywords that modify
    XML do not write those changes back to disk even if the source would be
    given as a path to a fike. Changes must always saved explicitly using
    `Save XML` keyword.

    When the source is given as a path to a file, the forward slash character
    (``/``) can be used as the path separator regardless the operating system.
    On Windows also the backslash works, but it the test data it needs to be
    escaped by doubling it (``\\\\``). Using the built-in variable ``${/}``
    naturally works too.

    = Example =

    The following simple example demonstrates parsing XML and verifying its
    contents both using keywords in this library and in _BuiltIn_ and
    _Collections_ libraries. How to use xpath expressions to find elements
    and what attributes the returned elements contain are dislwssed, with
    more examples, in `Finding elements with xpath` and `Element attributes`
    sections.

    In this example, as well as in many other examples in this documentation,
    ``${XML}`` refers to the following example XML document. In practice
    ``${XML}`` could either be a path to an XML file or it could contain the XML
    itself.

    | <example>
    |   <first id="1">text</first>
    |   <second id="2">
    |     <child/>
    |   </second>
    |   <third>
    |     <child>more text</child>
    |     <second id="child"/>
    |     <child><grandchild/></child>
    |   </third>
    |   <html>
    |     <p>
    |       Text with <b>bold</b> and <i>italics</i>.
    |     </p>
    |   </html>
    | </example>

    | ${root} =                | `Parse XML`   | ${XML}  |       |             |
    | `Should Be Equal`        | ${root.tag}   | example |       |             |
    | ${first} =               | `Get Element` | ${root} | first |             |
    | `Should Be Equal`        | ${first.text} | text    |       |             |
    | `Dictionary Should Contain Key` | ${first.attrib}  | id    |             |
    | `Element Text Should Be` | ${first}      | text    |       |             |
    | `Element Attribute Should Be` | ${first} | id      | 1     |             |
    | `Element Attribute Should Be` | ${root}  | id      | 1     | xpath=first |
    | `Element Attribute Should Be` | ${XML}   | id      | 1     | xpath=first |

    Notice that in the example three last lines are equivalent. Which one to
    use in practice depends on which other elements you need to get or verify.
    If you only need to do one verification, using the last line alone would
    suffice. If more verifications are needed, parsing the XML with `Parse XML`
    only once would be more efficient.

    = Finding elements with xpath =

    ElementTree, and thus also this library, supports finding elements using
    xpath expressions. ElementTree does not, however, support the full xpath
    syntax, and what is supported depends on its version. ElementTree 1.3 that
    is distributed with Python 2.7 supports richer syntax than earlier versions.

    The supported xpath syntax is explained below and
    [http://effbot.org/zone/element-xpath.htm|ElementTree documentation]
    provides more details. In the examples ``${XML}`` refers to the same XML
    structure as in the earlier example.

    If lxml support is enabled when `importing` the library, the whole
    [http://www.w3.org/TR/xpath/|xpath 1.0 standard] is supported.
    That includes everything listed below but also lot of other useful
    constructs.

    == Tag names ==

    When just a single tag name is used, xpath matches all direct child
    elements that have that tag name.

    | ${elem} =          | `Get Element`  | ${XML}      | third |
    | `Should Be Equal`  | ${elem.tag}    | third       |       |
    | @{children} =      | `Get Elements` | ${elem}     | child |
    | `Length Should Be` | ${children}    | 2           |       |

    == Paths ==

    Paths are created by combining tag names with a forward slash (``/``). For
    example, ``parent/child`` matches all ``child`` elements under ``parent``
    element. Notice that if there are multiple ``parent`` elements that all
    have ``child`` elements, ``parent/child`` xpath will match all these
    ``child`` elements.

    | ${elem} =         | `Get Element` | ${XML}     | second/child            |
    | `Should Be Equal` | ${elem.tag}   | child      |                         |
    | ${elem} =         | `Get Element` | ${XML}     | third/child/grandchild  |
    | `Should Be Equal` | ${elem.tag}   | grandchild |                         |

    == Wildcards ==

    An asterisk (``*``) can be used in paths instead of a tag name to denote
    any element.

    | @{children} =      | `Get Elements` | ${XML} | */child |
    | `Length Should Be` | ${children}    | 3      |         |

    == Current element ==

    The current element is denoted with a dot (``.``). Normally the current
    element is implicit and does not need to be included in the xpath.

    == Parent element ==

    The parent element of another element is denoted with two dots (``..``).
    Notice that it is not possible to refer to the parent of the current
    element. This syntax is supported only in ElementTree 1.3 (i.e.
    Python/Jython 2.7 and newer).

    | ${elem} =         | `Get Element` | ${XML} | */second/.. |
    | `Should Be Equal` | ${elem.tag}   | third  |             |

    == Search all sub elements ==

    Two forward slashes (``//``) mean that all sub elements, not only the
    direct children, are searched. If the search is started from the current
    element, an explicit dot is required.

    | @{elements} =      | `Get Elements` | ${XML} | .//second |
    | `Length Should Be` | ${elements}    | 2      |           |
    | ${b} =             | `Get Element`  | ${XML} | html//b   |
    | `Should Be Equal`  | ${b.text}      | bold   |           |

    == Predicates ==

    Predicates allow selecting elements using also other criteria than tag
    names, for example, attributes or position. They are specified after the
    normal tag name or path using syntax ``path[predicate]``. The path can have
    wildcards and other special syntax explained above.

    What predicates ElementTree supports is explained in the table below.
    Notice that predicates in general are supported only in ElementTree 1.3
    (i.e. Python/Jython 2.7 and newer).

    |  = Predicate =  |             = Matches =           |    = Example =     |
    | @attrib         | Elements with attribute ``attrib``. | second[@id]        |
    | @attrib="value" | Elements with attribute ``attrib`` having value ``value``. | *[@id="2"] |
    | position        | Elements at the specified position. Position can be an integer (starting from 1), expression ``last()``, or relative expression like ``last() - 1``. | third/child[1] |
    | tag             | Elements with a child element named ``tag``. | third/child[grandchild] |

    Predicates can also be stacked like ``path[predicate1][predicate2]``.
    A limitation is that possible position predicate must always be first.

    = Element attributes =

    All keywords returning elements, such as `Parse XML`, and `Get Element`,
    return ElementTree's
    [http://docs.python.org/library/xml.etree.elementtree.html#xml.etree.ElementTree.Element|Element objects].
    These elements can be used as inputs for other keywords, but they also
    contain several useful attributes that can be accessed directly using
    the extended variable syntax.

    The attributes that are both useful and colwenient to use in the test
    data are explained below. Also other attributes, including methods, can
    be accessed, but that is typically better to do in custom libraries than
    directly in the test data.

    The examples use the same ``${XML}`` structure as the earlier examples.

    == tag ==

    The tag of the element.

    | ${root} =         | `Parse XML` | ${XML}  |
    | `Should Be Equal` | ${root.tag} | example |

    == text ==

    The text that the element contains or Python ``None`` if the element has no
    text. Notice that the text _does not_ contain texts of possible child
    elements nor text after or between children. Notice also that in XML
    whitespace is significant, so the text contains also possible indentation
    and newlines. To get also text of the possible children, optionally
    whitespace normalized, use `Get Element Text` keyword.

    | ${1st} =          | `Get Element` | ${XML}  | first        |
    | `Should Be Equal` | ${1st.text}   | text    |              |
    | ${2nd} =          | `Get Element` | ${XML}  | second/child |
    | `Should Be Equal` | ${2nd.text}   | ${NONE} |              |
    | ${p} =            | `Get Element` | ${XML}  | html/p       |
    | `Should Be Equal` | ${p.text}     | \\n${SPACE*6}Text with${SPACE} |

    == tail ==

    The text after the element before the next opening or closing tag. Python
    ``None`` if the element has no tail. Similarly as with ``text``, also
    ``tail`` contains possible indentation and newlines.

    | ${b} =            | `Get Element` | ${XML}  | html/p/b  |
    | `Should Be Equal` | ${b.tail}     | ${SPACE}and${SPACE} |

    == attrib ==

    A Python dictionary containing attributes of the element.

    | ${2nd} =          | `Get Element`       | ${XML} | second |
    | `Should Be Equal` | ${2nd.attrib['id']} | 2      |        |
    | ${3rd} =          | `Get Element`       | ${XML} | third  |
    | `Should Be Empty` | ${3rd.attrib}       |        |        |

    = Handling XML namespaces =

    ElementTree and lxml handle possible namespaces in XML dolwments by adding
    the namespace URI to tag names in so called Clark Notation. That is
    incolwenient especially with xpaths, and by default this library strips
    those namespaces away and moves them to ``xmlns`` attribute instead. That
    can be avoided by passing ``keep_clark_notation`` argument to `Parse XML`
    keyword. The pros and cons of both approaches are dislwssed in more detail
    below.

    == How ElementTree handles namespaces ==

    If an XML document has namespaces, ElementTree adds namespace information
    to tag names in [http://www.jclark.com/xml/xmlns.htm|Clark Notation]
    (e.g. ``{http://ns.uri}tag``) and removes original ``xmlns`` attributes.
    This is done both with default namespaces and with namespaces with a prefix.
    How it works in practice is illustrated by the following example, where
    ``${NS}`` variable contains this XML document:

    | <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    |                 xmlns="http://www.w3.org/1999/xhtml">
    |   <xsl:template match="/">
    |     <html></html>
    |   </xsl:template>
    | </xsl:stylesheet>

    | ${root} = | `Parse XML` | ${NS} | keep_clark_notation=yes |
    | `Should Be Equal` | ${root.tag} | {http://www.w3.org/1999/XSL/Transform}stylesheet |
    | `Element Should Exist` | ${root} | {http://www.w3.org/1999/XSL/Transform}template/{http://www.w3.org/1999/xhtml}html |
    | `Should Be Empty` | ${root.attrib} |

    As you can see, including the namespace URI in tag names makes xpaths
    really long and complex.

    If you save the XML, ElementTree moves namespace information back to
    ``xmlns`` attributes. Unfortunately it does not restore the original
    prefixes:

    | <ns0:stylesheet xmlns:ns0="http://www.w3.org/1999/XSL/Transform">
    |   <ns0:template match="/">
    |     <ns1:html xmlns:ns1="http://www.w3.org/1999/xhtml"></ns1:html>
    |   </ns0:template>
    | </ns0:stylesheet>

    The resulting output is semantically same as the original, but mangling
    prefixes like this may still not be desirable. Notice also that the actual
    output depends slightly on ElementTree version.

    == Default namespace handling ==

    Because the way ElementTree handles namespaces makes xpaths so complicated,
    this library, by default, strips namespaces from tag names and moves that
    information back to ``xmlns`` attributes. How this works in practice is
    shown by the example below, where ``${NS}`` variable contains the same XML
    document as in the previous example.

    | ${root} = | `Parse XML` | ${NS} |
    | `Should Be Equal` | ${root.tag} | stylesheet |
    | `Element Should Exist` | ${root} | template/html |
    | `Element Attribute Should Be` | ${root} | xmlns | http://www.w3.org/1999/XSL/Transform |
    | `Element Attribute Should Be` | ${root} | xmlns | http://www.w3.org/1999/xhtml | xpath=template/html |

    Now that tags do not contain namespace information, xpaths are simple again.

    A minor limitation of this approach is that namespace prefixes are lost.
    As a result the saved output is not exactly same as the original one in
    this case either:

    | <stylesheet xmlns="http://www.w3.org/1999/XSL/Transform">
    |   <template match="/">
    |     <html xmlns="http://www.w3.org/1999/xhtml"></html>
    |   </template>
    | </stylesheet>

    Also this output is semantically same as the original. If the original XML
    had only default namespaces, the output would also look identical.

    == Namespaces with lxml ==

    Namespaces are handled the same way also if lxml mode is enabled when
    `importing` the library. The only difference is that lxml stores information
    about namespace prefixes and thus they are preserved if XML is saved.

    == Attribute namespaces ==

    Attributes in XML dolwments are, by default, in the same namespaces as
    the element they belong to. It is possible to use different namespaces
    by using prefixes, but this is pretty rare.

    If an attribute has a namespace prefix, ElementTree will replace it with
    Clark Notation the same way it handles elements. Because stripping
    namespaces from attributes could cause attribute conflicts, this library
    does not handle attribute namespaces at all. Thus the following example
    works the same way regardless how namespaces are handled.

    | ${root} = | `Parse XML` | <root id="1" ns:id="2" xmlns:ns="http://my.ns"/> |
    | `Element Attribute Should Be` | ${root} | id | 1 |
    | `Element Attribute Should Be` | ${root} | {http://my.ns}id | 2 |
    """

    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = get_version()
    _whitespace = re.compile('\s+')
    _xml_declaration = re.compile('^<\?xml .*\?>\n')

    def __init__(self, use_lxml=False):
        """Import library with optionally lxml mode enabled.

        By default this library uses Python's standard
        [https://docs.python.org/2/library/xml.etree.elementtree.html|ElementTree]
        module for parsing XML. If ``use_lxml`` argument is given any true
        value (e.g. any non-empty string), the library will use
        [http://lxml.de|lxml] instead. See `introduction` for benefits
        provided by lxml.

        Using lxml requires that the lxml module is installed on the system.
        If lxml mode is enabled but the module is not installed, this library
        will emit a warning and revert back to using the standard ElementTree.

        The support for lxml is new in Robot Framework 2.8.5.
        """
        if use_lxml and lxml_etree:
            self.etree = lxml_etree
            self.modern_etree = True
            self.lxml_etree = True
        else:
            self.etree = ET
            self.modern_etree = ET.VERSION >= '1.3'
            self.lxml_etree = False
        if use_lxml and not lxml_etree:
            logger.warn('XML library reverted to use standard ElementTree '
                        'because lxml module is not installed.')

    def parse_xml(self, source, keep_clark_notation=False):
        """Parses the given XML file or string into an element structure.

        The ``source`` can either be a path to an XML file or a string
        containing XML. In both cases the XML is parsed into ElementTree
        [http://docs.python.org/library/xml.etree.elementtree.html#xml.etree.ElementTree.Element|element structure]
        and the root element is returned.

        As dislwssed in `Handling XML namespaces` section, this keyword, by
        default, strips possible namespaces added by ElementTree into tag names.
        This typically eases handling XML dolwments with namespaces
        considerably. If you do not want that to happen, or want to avoid
        the small overhead of going through the element structure when your
        XML does not have namespaces, you can disable this feature by giving
        ``keep_clark_notation`` argument a true value (e.g. any non-empty
        string).

        Examples:
        | ${root} = | Parse XML | <root><child/></root> |
        | ${xml} =  | Parse XML | ${LWRDIR}/test.xml    | no namespace cleanup |

        Use `Get Element` keyword if you want to get a certain element and not
        the whole structure. See `Parsing XML` section for more details and
        examples.

        Stripping namespaces is a new feature in Robot Framework 2.7.5.
        """
        with ETSource(source) as source:
            root = self.etree.parse(source).getroot()
        if self.lxml_etree:
            self._remove_comments(root)
        if not keep_clark_notation:
            NameSpaceStripper().strip(root)
        return root

    def _remove_comments(self, node):
        for comment in node.xpath('//comment()'):
            parent = comment.getparent()
            if parent:
                self._preserve_tail(comment, parent)
                parent.remove(comment)

    def get_element(self, source, xpath='.'):
        """Returns an element in the ``source`` matching the ``xpath``.

        The ``source`` can be a path to an XML file, a string containing XML, or
        an already parsed XML element. The ``xpath`` specifies which element to
        find. See the `introduction` for more details about both the possible
        sources and the supported xpath syntax.

        The keyword fails if more, or less, than one element matches the
        ``xpath``. Use `Get Elements` if you want all matching elements to be
        returned.

        Examples using ``${XML}`` structure from `Example`:
        | ${element} = | Get Element | ${XML}     | second |
        | ${child} =   | Get Element | ${element} | child  |

        `Parse XML` is recommended for parsing XML when the whole structure
        is needed. It must be used if there is a need to configure how XML
        namespaces are handled.

        Many other keywords use this keyword internally, and keywords modifying
        XML are typically dolwmented to both to modify the given source and
        to return it. Modifying the source does not apply if the source is
        given as a string. The XML structure parsed based on the string and
        then modified is nevertheless returned.
        """
        elements = self.get_elements(source, xpath)
        if len(elements) != 1:
            self._raise_wrong_number_of_matches(len(elements), xpath)
        return elements[0]

    def _raise_wrong_number_of_matches(self, count, xpath, message=None):
        if not message:
            message = self._wrong_number_of_matches(count, xpath)
        raise AssertionError(message)

    def _wrong_number_of_matches(self, count, xpath):
        if not count:
            return "No element matching '%s' found." % xpath
        if count == 1:
            return "One element matching '%s' found." % xpath
        return "Multiple elements (%d) matching '%s' found." % (count, xpath)

    def get_elements(self, source, xpath):
        """Returns a list of elements in the ``source`` matching the ``xpath``.

        The ``source`` can be a path to an XML file, a string containing XML, or
        an already parsed XML element. The ``xpath`` specifies which element to
        find. See the `introduction` for more details.

        Elements matching the ``xpath`` are returned as a list. If no elements
        match, an empty list is returned. Use `Get Element` if you want to get
        exactly one match.

        Examples using ``${XML}`` structure from `Example`:
        | ${children} =    | Get Elements | ${XML} | third/child |
        | Length Should Be | ${children}  | 2      |             |
        | ${children} =    | Get Elements | ${XML} | first/child |
        | Should Be Empty  |  ${children} |        |             |
        """
        if isinstance(source, basestring):
            source = self.parse_xml(source)
        finder = ElementFinder(self.etree, self.modern_etree, self.lxml_etree)
        return finder.find_all(source, xpath)

    def get_child_elements(self, source, xpath='.'):
        """Returns the child elements of the specified element as a list.

        The element whose children to return is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        All the direct child elements of the specified element are returned.
        If the element has no children, an empty list is returned.

        Examples using ``${XML}`` structure from `Example`:
        | ${children} =    | Get Child Elements | ${XML} |             |
        | Length Should Be | ${children}        | 4      |             |
        | ${children} =    | Get Child Elements | ${XML} | xpath=first |
        | Should Be Empty  | ${children}        |        |             |
        """
        return list(self.get_element(source, xpath))

    def get_element_count(self, source, xpath='.'):
        """Returns and logs how many elements the given ``xpath`` matches.

        Arguments ``source`` and ``xpath`` have exactly the same semantics as
        with `Get Elements` keyword that this keyword uses internally.

        See also `Element Should Exist` and `Element Should Not Exist`.

        New in Robot Framework 2.7.5.
        """
        count = len(self.get_elements(source, xpath))
        logger.info("%d element%s matched '%s'." % (count, s(count), xpath))
        return count

    def element_should_exist(self, source, xpath='.', message=None):
        """Verifies that one or more element match the given ``xpath``.

        Arguments ``source`` and ``xpath`` have exactly the same semantics as
        with `Get Elements` keyword. Keyword passes if the ``xpath`` matches
        one or more elements in the ``source``. The default error message can
        be overridden with the ``message`` argument.

        See also `Element Should Not Exist` as well as `Get Element Count`
        that this keyword uses internally.

        New in Robot Framework 2.7.5.
        """
        count = self.get_element_count(source, xpath)
        if not count:
            self._raise_wrong_number_of_matches(count, xpath, message)

    def element_should_not_exist(self, source, xpath='.', message=None):
        """Verifies that no element match the given ``xpath``.

        Arguments ``source`` and ``xpath`` have exactly the same semantics as
        with `Get Elements` keyword. Keyword fails if the ``xpath`` matches any
        element in the ``source``. The default error message can be overridden
        with the ``message`` argument.

        See also `Element Should Exist` as well as `Get Element Count`
        that this keyword uses internally.

        New in Robot Framework 2.7.5.
        """
        count = self.get_element_count(source, xpath)
        if count:
            self._raise_wrong_number_of_matches(count, xpath, message)

    def get_element_text(self, source, xpath='.', normalize_whitespace=False):
        """Returns all text of the element, possibly whitespace normalized.

        The element whose text to return is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        This keyword returns all the text of the specified element, including
        all the text its children and grandchildren contains. If the element
        has no text, an empty string is returned. The returned text is thus not
        always the same as the `text` attribute of the element.

        Be default all whitespace, including newlines and indentation, inside
        the element is returned as-is. If ``normalize_whitespace`` is given any
        true value (e.g. any non-empty string), then leading and trailing
        whitespace is stripped, newlines and tabs colwerted to spaces, and
        multiple spaces collapsed into one. This is especially useful when
        dealing with HTML data.

        Examples using ``${XML}`` structure from `Example`:
        | ${text} =       | Get Element Text | ${XML}       | first        |
        | Should Be Equal | ${text}          | text         |              |
        | ${text} =       | Get Element Text | ${XML}       | second/child |
        | Should Be Empty | ${text}          |              |              |
        | ${paragraph} =  | Get Element      | ${XML}       | html/p       |
        | ${text} =       | Get Element Text | ${paragraph} | normalize_whitespace=yes |
        | Should Be Equal | ${text}          | Text with bold and italics. |

        See also `Get Elements Texts`, `Element Text Should Be` and
        `Element Text Should Match`.
        """
        element = self.get_element(source, xpath)
        text = ''.join(self._yield_texts(element))
        if normalize_whitespace:
            text = self._normalize_whitespace(text)
        return text

    def _yield_texts(self, element, top=True):
        if element.text:
            yield element.text
        for child in element:
            for text in self._yield_texts(child, top=False):
                yield text
        if element.tail and not top:
            yield element.tail

    def _normalize_whitespace(self, text):
        return self._whitespace.sub(' ', text.strip())

    def get_elements_texts(self, source, xpath, normalize_whitespace=False):
        """Returns text of all elements matching ``xpath`` as a list.

        The elements whose text to return is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Elements`
        keyword.

        The text of the matched elements is returned using the same logic
        as with `Get Element Text`. This includes optional whitespace
        normalization using the ``normalize_whitespace`` option.

        Examples using ``${XML}`` structure from `Example`:
        | @{texts} =       | Get Elements Texts | ${XML}    | third/child |
        | Length Should Be | ${texts}           | 2         |             |
        | Should Be Equal  | @{texts}[0]        | more text |             |
        | Should Be Equal  | @{texts}[1]        | ${EMPTY}  |             |
        """
        return [self.get_element_text(elem, normalize_whitespace=normalize_whitespace)
                for elem in self.get_elements(source, xpath)]

    def element_text_should_be(self, source, expected, xpath='.',
                               normalize_whitespace=False, message=None):
        """Verifies that the text of the specified element is ``expected``.

        The element whose text is verified is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        The text to verify is got from the specified element using the same
        logic as with `Get Element Text`. This includes optional whitespace
        normalization using the ``normalize_whitespace`` option.

        The keyword passes if the text of the element is equal to the
        ``expected`` value, and otherwise it fails. The default error message
        can be overridden with the ``message`` argument.  Use `Element Text
        Should Match` to verify the text against a pattern instead of an exact
        value.

        Examples using ``${XML}`` structure from `Example`:
        | Element Text Should Be | ${XML}       | text     | xpath=first      |
        | Element Text Should Be | ${XML}       | ${EMPTY} | xpath=second/child |
        | ${paragraph} =         | Get Element  | ${XML}   | xpath=html/p     |
        | Element Text Should Be | ${paragraph} | Text with bold and italics. | normalize_whitespace=yes |
        """
        text = self.get_element_text(source, xpath, normalize_whitespace)
        should_be_equal(text, expected, message, values=False)

    def element_text_should_match(self, source, pattern, xpath='.',
                                  normalize_whitespace=False, message=None):
        """Verifies that the text of the specified element matches ``expected``.

        This keyword works exactly like `Element Text Should Be` except that
        the expected value can be given as a pattern that the text of the
        element must match.

        Pattern matching is similar as matching files in a shell, and it is
        always case-sensitive. In the pattern, '*' matches anything and '?'
        matches any single character.

        Examples using ``${XML}`` structure from `Example`:
        | Element Text Should Match | ${XML}       | t???   | xpath=first  |
        | ${paragraph} =            | Get Element  | ${XML} | xpath=html/p |
        | Element Text Should Match | ${paragraph} | Text with * and *. | normalize_whitespace=yes |
        """
        text = self.get_element_text(source, xpath, normalize_whitespace)
        should_match(text, pattern, message, values=False)

    def get_element_attribute(self, source, name, xpath='.', default=None):
        """Returns the named attribute of the specified element.

        The element whose attribute to return is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        The value of the attribute ``name`` of the specified element is returned.
        If the element does not have such element, the ``default`` value is
        returned instead.

        Examples using ``${XML}`` structure from `Example`:
        | ${attribute} =  | Get Element Attribute | ${XML} | id | xpath=first |
        | Should Be Equal | ${attribute}          | 1      |    |             |
        | ${attribute} =  | Get Element Attribute | ${XML} | xx | xpath=first | default=value |
        | Should Be Equal | ${attribute}          | value  |    |             |

        See also `Get Element Attributes`, `Element Attribute Should Be`,
        `Element Attribute Should Match` and `Element Should Not Have Attribute`.
        """
        return self.get_element(source, xpath).get(name, default)

    def get_element_attributes(self, source, xpath='.'):
        """Returns all attributes of the specified element.

        The element whose attributes to return is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        Attributes are returned as a Python dictionary. It is a copy of the
        original attributes so modifying it has no effect on the XML structure.

        Examples using ``${XML}`` structure from `Example`:
        | ${attributes} = | Get Element Attributes      | ${XML} | first |
        | Dictionary Should Contain Key | ${attributes} | id     |       |
        | ${attributes} = | Get Element Attributes      | ${XML} | third |
        | Should Be Empty | ${attributes}               |        |       |

        Use `Get Element Attribute` to get the value of a single attribute.
        """
        return dict(self.get_element(source, xpath).attrib)

    def element_attribute_should_be(self, source, name, expected, xpath='.',
                                    message=None):
        """Verifies that the specified attribute is ``expected``.

        The element whose attribute is verified is specified using ``source``
        and ``xpath``. They have exactly the same semantics as with
        `Get Element` keyword.

        The keyword passes if the attribute ``name`` of the element is equal to
        the ``expected`` value, and otherwise it fails. The default error
        message can be overridden with the ``message`` argument.

        To test that the element does not have a certain attribute, Python
        ``None`` (i.e. variable ``${NONE}``) can be used as the expected value.
        A cleaner alternative is using `Element Should Not Have Attribute`.

        Examples using ``${XML}`` structure from `Example`:
        | Element Attribute Should Be | ${XML} | id | 1       | xpath=first |
        | Element Attribute Should Be | ${XML} | id | ${NONE} |             |

        See also `Element Attribute Should Match` and `Get Element Attribute`.
        """
        attr = self.get_element_attribute(source, name, xpath)
        should_be_equal(attr, expected, message, values=False)

    def element_attribute_should_match(self, source, name, pattern, xpath='.',
                                       message=None):
        """Verifies that the specified attribute matches ``expected``.

        This keyword works exactly like `Element Attribute Should Be` except
        that the expected value can be given as a pattern that the attribute of
        the element must match.

        Pattern matching is similar as matching files in a shell, and it is
        always case-sensitive. In the pattern, '*' matches anything and '?'
        matches any single character.

        Examples using ``${XML}`` structure from `Example`:
        | Element Attribute Should Match | ${XML} | id | ?   | xpath=first |
        | Element Attribute Should Match | ${XML} | id | c*d | xpath=third/second |
        """
        attr = self.get_element_attribute(source, name, xpath)
        if attr is None:
            raise AssertionError("Attribute '%s' does not exist." % name)
        should_match(attr, pattern, message, values=False)

    def element_should_not_have_attribute(self, source, name, xpath='.', message=None):
        """Verifies that the specified element does not have  attribute ``name``.

        The element whose attribute is verified is specified using ``source``
        and ``xpath``. They have exactly the same semantics as with
        `Get Element` keyword.

        The keyword fails if the specified element has attribute ``name``. The
        default error message can be overridden with the ``message`` argument.

        Examples using ``${XML}`` structure from `Example`:
        | Element Should Not Have Attribute | ${XML} | id  |
        | Element Should Not Have Attribute | ${XML} | xxx | xpath=first |

        See also `Get Element Attribute`, `Get Element Attributes`,
        `Element Text Should Be` and `Element Text Should Match`.

        New in Robot Framework 2.7.5.
        """
        attr = self.get_element_attribute(source, name, xpath)
        if attr is not None:
            raise AssertionError(message or "Attribute '%s' exists and "
                                            "has value '%s'." % (name, attr))

    def elements_should_be_equal(self, source, expected, exclude_children=False,
                                 normalize_whitespace=False):
        """Verifies that the given ``source`` element is equal to ``expected``.

        Both ``source`` and ``expected`` can be given as a path to an XML file,
        as a string containing XML, or as an already parsed XML element
        structure. See `introduction` for more information about parsing XML in
        general.

        The keyword passes if the ``source`` element and ``expected`` element
        are equal. This includes testing the tag names, texts, and attributes
        of the elements. By default also child elements are verified the same
        way, but this can be disabled by setting ``exclude_children`` to any
        true value (e.g. any non-empty string).

        All texts inside the given elements are verified, but possible text
        outside them is not. By default texts must match exactly, but setting
        ``normalize_whitespace`` to any true value makes text verification
        independent on newlines, tabs, and the amount of spaces. For more
        details about handling text see `Get Element Text` keyword and
        dislwssion about elements' `text` and `tail` attributes in the
        `introduction`.

        Examples using ``${XML}`` structure from `Example`:
        | ${first} =               | Get Element | ${XML} | first             |
        | Elements Should Be Equal | ${first}    | <first id="1">text</first> |
        | ${p} =                   | Get Element | ${XML} | html/p            |
        | Elements Should Be Equal | ${p} | <p>Text with <b>bold</b> and <i>italics</i>.</p> | normalize_whitespace=yes |
        | Elements Should Be Equal | ${p} | <p>Text with</p> | exclude | normalize |

        The last example may look a bit strange because the ``<p>`` element only
        has text ``Text with``. The reason is that rest of the text inside
        ``<p>`` actually belongs to the child elements.

        See also `Elements Should Match`.
        """
        self._compare_elements(source, expected, should_be_equal,
                               exclude_children, normalize_whitespace)

    def elements_should_match(self, source, expected, exclude_children=False,
                              normalize_whitespace=False):
        """Verifies that the given ``source`` element matches ``expected``.

        This keyword works exactly like `Elements Should Be Equal` except that
        texts and attribute values in the expected value can be given as
        patterns.

        Pattern matching is similar as matching files in a shell, and it is
        always case-sensitive. In the pattern, '*' matches anything and '?'
        matches any single character.

        Examples using ``${XML}`` structure from `Example`:
        | ${first} =            | Get Element | ${XML} | first          |
        | Elements Should Match | ${first}    | <first id="?">*</first> |

        See `Elements Should Be Equal` for more examples.
        """
        self._compare_elements(source, expected, should_match,
                               exclude_children, normalize_whitespace)

    def _compare_elements(self, source, expected, comparator, exclude_children,
                          normalize_whitespace):
        normalizer = self._normalize_whitespace if normalize_whitespace else None
        comparator = ElementComparator(comparator, normalizer, exclude_children)
        comparator.compare(self.get_element(source), self.get_element(expected))

    def set_element_tag(self, source, tag, xpath='.'):
        """Sets the tag of the specified element.

        The element whose tag to set is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        Examples using ``${XML}`` structure from `Example`:
        | Set Element Tag      | ${XML}     | newTag     |
        | Should Be Equal      | ${XML.tag} | newTag     |
        | Set Element Tag      | ${XML}     | xxx        | xpath=second/child |
        | Element Should Exist | ${XML}     | second/xxx |
        | Element Should Not Exist | ${XML} | second/child |

        Can only set the tag of a single element. Use `Set Elements Tag` to set
        the tag of multiple elements in one call.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        self.get_element(source, xpath).tag = tag
        return source

    def set_elements_tag(self, source, tag, xpath='.'):
        """Sets the tag of the specified elements.

        Like `Set Element Tag` but sets the tag of all elements matching
        the given ``xpath``.

        New in Robot Framework 2.8.6.
        """
        for elem in self.get_elements(source, xpath):
            self.set_element_tag(elem, tag)

    def set_element_text(self, source, text=None, tail=None, xpath='.'):
        """Sets text and/or tail text of the specified element.

        The element whose text to set is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        Element's text and tail text are changed only if new ``text`` and/or
        ``tail`` values are given. See `Element attributes` section for more
        information about `text` and `tail` in general.

        Examples using ``${XML}`` structure from `Example`:
        | Set Element Text       | ${XML} | new text | xpath=first    |
        | Element Text Should Be | ${XML} | new text | xpath=first    |
        | Set Element Text       | ${XML} | tail=&   | xpath=html/p/b |
        | Element Text Should Be | ${XML} | Text with bold&italics. | xpath=html/p  | normalize_whitespace=yes |
        | Set Element Text       | ${XML} | slanted  | !! | xpath=html/p/i |
        | Element Text Should Be | ${XML} | Text with bold&slanted!! | xpath=html/p  | normalize_whitespace=yes |

        Can only set the text/tail of a single element. Use `Set Elements Text`
        to set the text/tail of multiple elements in one call.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        element = self.get_element(source, xpath)
        if text is not None:
            element.text = text
        if tail is not None:
            element.tail = tail
        return source

    def set_elements_text(self, source, text=None, tail=None, xpath='.'):
        """Sets text and/or tail text of the specified elements.

        Like `Set Element Text` but sets the text or tail of all elements
        matching the given ``xpath``.

        New in Robot Framework 2.8.6.
        """
        for elem in self.get_elements(source, xpath):
            self.set_element_text(elem, text, tail)

    def set_element_attribute(self, source, name, value, xpath='.'):
        """Sets attribute ``name`` of the specified element to ``value``.

        The element whose attribute to set is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        It is possible to both set new attributes and to overwrite existing.
        Use `Remove Element Attribute` or `Remove Element Attributes` for
        removing them.

        Examples using ``${XML}`` structure from `Example`:
        | Set Element Attribute       | ${XML} | attr | value |
        | Element Attribute Should Be | ${XML} | attr | value |
        | Set Element Attribute       | ${XML} | id   | new   | xpath=first |
        | Element Attribute Should Be | ${XML} | id   | new   | xpath=first |

        Can only set an attribute of a single element. Use `Set Elements
        Attribute` to set an attribute of multiple elements in one call.

        New in Robot Framework 2.7.5.
        """
        if not name:
            raise RuntimeError('Attribute name can not be empty.')
        source = self.get_element(source)
        self.get_element(source, xpath).attrib[name] = value
        return source

    def set_elements_attribute(self, source, name, value, xpath='.'):
        """Sets attribute ``name`` of the specified elements to ``value``.

        Like `Set Element Attribute` but sets the attribute of all elements
        matching the given ``xpath``.

        New in Robot Framework 2.8.6.
        """
        for elem in self.get_elements(source, xpath):
            self.set_element_attribute(elem, name, value)

    def remove_element_attribute(self, source, name, xpath='.'):
        """Removes attribute ``name`` from the specified element.

        The element whose attribute to remove is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        It is not a failure to remove a non-existing attribute. Use `Remove
        Element Attributes` to remove all attributes and `Set Element Attribute`
        to set them.

        Examples using ``${XML}`` structure from `Example`:
        | Remove Element Attribute          | ${XML} | id | xpath=first |
        | Element Should Not Have Attribute | ${XML} | id | xpath=first |

        Can only remove an attribute from a single element. Use `Remove Elements
        Attribute` to remove an attribute of multiple elements in one call.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        attrib = self.get_element(source, xpath).attrib
        if name in attrib:
            attrib.pop(name)
        return source

    def remove_elements_attribute(self, source, name, xpath='.'):
        """Removes attribute ``name`` from the specified elements.

        Like `Remove Element Attribute` but removes the attribute of all
        elements matching the given ``xpath``.

        New in Robot Framework 2.8.6.
        """
        for elem in self.get_elements(source, xpath):
            self.remove_element_attribute(elem, name)

    def remove_element_attributes(self, source, xpath='.'):
        """Removes all attributes from the specified element.

        The element whose attributes to remove is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        Use `Remove Element Attribute` to remove a single attribute and
        `Set Element Attribute` to set them.

        Examples using ``${XML}`` structure from `Example`:
        | Remove Element Attributes         | ${XML} | xpath=first |
        | Element Should Not Have Attribute | ${XML} | id | xpath=first |

        Can only remove attributes from a single element. Use `Remove Elements
        Attributes` to remove all attributes of multiple elements in one call.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        self.get_element(source, xpath).attrib.clear()
        return source

    def remove_elements_attributes(self, source, xpath='.'):
        """Removes all attributes from the specified elements.

        Like `Remove Element Attributes` but removes all attributes of all
        elements matching the given ``xpath``.

        New in Robot Framework 2.8.6.
        """
        for elem in self.get_elements(source, xpath):
            self.remove_element_attributes(elem)

    def add_element(self, source, element, index=None, xpath='.'):
        """Adds a child element to the specified element.

        The element to whom to add the new element is specified using ``source``
        and ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword. The resulting XML structure is returned, and if the ``source``
        is an already parsed XML structure, it is also modified in place.

        The ``element`` to add can be specified as a path to an XML file or
        as a string containing XML, or it can be an already parsed XML element.
        The element is copied before adding so modifying either the original
        or the added element has no effect on the other
        .
        The element is added as the last child by default, but a custom index
        can be used to alter the position. Indices start from zero (0 = first
        position, 1 = second position, etc.), and negative numbers refer to
        positions at the end (-1 = second last position, -2 = third last, etc.).

        Examples using ``${XML}`` structure from `Example`:
        | Add Element | ${XML} | <new id="x"><c1/></new> |
        | Add Element | ${XML} | <c2/> | xpath=new |
        | Add Element | ${XML} | <c3/> | index=1 | xpath=new |
        | ${new} = | Get Element | ${XML} | new |
        | Elements Should Be Equal | ${new} | <new id="x"><c1/><c3/><c2/></new> |

        Use `Remove Element` or `Remove Elements` to remove elements.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        parent = self.get_element(source, xpath)
        element = self.copy_element(element)
        if index is None:
            parent.append(element)
        else:
            parent.insert(int(index), element)
        return source

    def remove_element(self, source, xpath='', remove_tail=False):
        """Removes the element matching ``xpath`` from the ``source`` structure.

        The element to remove from the ``source`` is specified with ``xpath``
        using the same semantics as with `Get Element` keyword. The resulting
        XML structure is returned, and if the ``source`` is an already parsed
        XML structure, it is also modified in place.

        The keyword fails if ``xpath`` does not match exactly one element.
        Use `Remove Elements` to remove all matched elements.

        Element's tail text is not removed by default, but that can be changed
        by giving ``remove_tail`` a true value (e.g. any non-empty string).
        See ``Element attributes`` section for more information about `tail` in
        general.

        Examples using ``${XML}`` structure from `Example`:
        | Remove Element           | ${XML} | xpath=second |
        | Element Should Not Exist | ${XML} | xpath=second |
        | Remove Element           | ${XML} | xpath=html/p/b | remove_tail=yes |
        | Element Text Should Be   | ${XML} | Text with italics. | xpath=html/p | normalize_whitespace=yes |

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        self._remove_element(source, self.get_element(source, xpath), remove_tail)
        return source

    def remove_elements(self, source, xpath='', remove_tail=False):
        """Removes all elements matching ``xpath`` from the ``source`` structure.

        The elements to remove from the ``source`` are specified with ``xpath``
        using the same semantics as with `Get Elements` keyword. The resulting
        XML structure is returned, and if the ``source`` is an already parsed
        XML structure, it is also modified in place.

        It is not a failure if ``xpath`` matches no elements. Use `Remove
        Element` to remove exactly one element.

        Element's tail text is not removed by default, but that can be changed
        by using ``remove_tail`` argument similarly as with `Remove Element`.

        Examples using ``${XML}`` structure from `Example`:
        | Remove Elements          | ${XML} | xpath=*/child      |
        | Element Should Not Exist | ${XML} | xpath=second/child |
        | Element Should Not Exist | ${XML} | xpath=third/child  |

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        for element in self.get_elements(source, xpath):
            self._remove_element(source, element, remove_tail)
        return source

    def _remove_element(self, root, element, remove_tail=False):
        parent = self._find_parent(root, element)
        if not remove_tail:
            self._preserve_tail(element, parent)
        parent.remove(element)

    def _find_parent(self, root, element):
        for parent in root.getiterator():
            for child in parent:
                if child is element:
                    return parent
        raise RuntimeError('Cannot remove root element.')

    def _preserve_tail(self, element, parent):
        if not element.tail:
            return
        index = list(parent).index(element)
        if index == 0:
            parent.text = (parent.text or '') + element.tail
        else:
            sibling = parent[index-1]
            sibling.tail = (sibling.tail or '') + element.tail

    def clear_element(self, source, xpath='.', clear_tail=False):
        """Clears the contents of the specified element.

        The element to clear is specified using ``source`` and ``xpath``. They
        have exactly the same semantics as with `Get Element` keyword.
        The resulting XML structure is returned, and if the ``source`` is
        an already parsed XML structure, it is also modified in place.

        Clearing the element means removing its text, attributes, and children.
        Element's tail text is not removed by default, but that can be changed
        by giving ``clear_tail`` a true value (e.g. any non-empty string).
        See `Element attributes` section for more information about tail in
        general.

        Examples using ``${XML}`` structure from `Example`:
        | Clear Element            | ${XML}   | xpath=first |
        | ${first} = | Get Element | ${XML}   | xpath=first |
        | Elements Should Be Equal | ${first} | <first/>    |
        | Clear Element            | ${XML}   | xpath=html/p/b | clear_tail=yes |
        | Element Text Should Be   | ${XML}   | Text with italics. | xpath=html/p | normalize_whitespace=yes |
        | Clear Element            | ${XML}   |
        | Elements Should Be Equal | ${XML}   | <example/> |

        Use `Remove Element` to remove the whole element.

        New in Robot Framework 2.7.5.
        """
        source = self.get_element(source)
        element = self.get_element(source, xpath)
        tail = element.tail
        element.clear()
        if not clear_tail:
            element.tail = tail
        return source

    def copy_element(self, source, xpath='.'):
        """Returns a copy of the specified element.

        The element to copy is specified using ``source`` and ``xpath``. They
        have exactly the same semantics as with `Get Element` keyword.

        If the copy or the original element is modified afterwards, the changes
        have no effect on the other.

        Examples using ``${XML}`` structure from `Example`:
        | ${elem} =  | Get Element  | ${XML}  | xpath=first |
        | ${copy1} = | Copy Element | ${elem} |
        | ${copy2} = | Copy Element | ${XML}  | xpath=first |
        | Set Element Text         | ${XML}   | new text    | xpath=first      |
        | Set Element Attribute    | ${copy1} | id          | new              |
        | Elements Should Be Equal | ${elem}  | <first id="1">new text</first> |
        | Elements Should Be Equal | ${copy1} | <first id="new">text</first>   |
        | Elements Should Be Equal | ${copy2} | <first id="1">text</first>     |

        New in Robot Framework 2.7.5.
        """
        return copy.deepcopy(self.get_element(source, xpath))

    def element_to_string(self, source, xpath='.'):
        """Returns the string representation of the specified element.

        The element to colwert to a string is specified using ``source`` and
        ``xpath``. They have exactly the same semantics as with `Get Element`
        keyword.

        The returned string is in Unicode format and it does not contain any
        XML declaration.

        See also `Log Element` and `Save XML`.
        """
        string = self.etree.tostring(self.get_element(source, xpath), encoding='UTF-8')
        return self._xml_declaration.sub('', string.decode('UTF-8')).strip()

    def log_element(self, source, level='INFO', xpath='.'):
        """Logs the string representation of the specified element.

        The element specified with ``source`` and ``xpath`` is first colwerted
        into a string using `Element To String` keyword internally. The
        resulting string is then logged using the given ``level``.

        The logged string is also returned.
        """
        string = self.element_to_string(source, xpath)
        logger.write(string, level)
        return string

    def save_xml(self, source, path, encoding='UTF-8'):
        """Saves the given element to the specified file.

        The element to save is specified with ``source`` using the same
        semantics as with `Get Element` keyword.

        The file where the element is saved is denoted with ``path`` and the
        encoding to use with ``encoding``. The resulting file contains an XML
        declaration.

        Use `Element To String` if you just need a string representation of
        the element,

        New in Robot Framework 2.7.5.
        """
        elem = self.get_element(source)
        if self.lxml_etree:
            NameSpaceStripper().unstrip(elem)
        tree = self.etree.ElementTree(elem)
        xml_declaration = {'xml_declaration': True} if self.modern_etree else {}
        # Need to explicitly open/close files because older ET versions don't
        # close files they open and Jython/IPY don't close them implicitly.
        with open(path, 'w') as output:
            tree.write(output, encoding=encoding, **xml_declaration)

    def evaluate_xpath(self, source, expression, context='.'):
        """Evaluates the given xpath expression and returns results.

        The element in which context the expression is exelwted is specified
        using ``source`` and ``context`` arguments. They have exactly the same
        semantics as ``source`` and ``xpath`` arguments have with `Get Element`
        keyword.

        The xpath expression to evaluate is given as ``expression`` argument.
        The result of the evaluation is returned as-is.

        Examples using ``${XML}`` structure from `Example`:
        | ${count} =      | Evaluate Xpath | ${XML}  | count(third/*) |
        | Should Be Equal | ${count}       | ${3}    |
        | ${text} =       | Evaluate Xpath | ${XML}  | string(descendant::second[last()]/@id) |
        | Should Be Equal | ${text}        | child   |
        | ${bold} =       | Evaluate Xpath | ${XML}  | boolean(preceding-sibling::*[1] = 'bold') | context=html/p/i |
        | Should Be Equal | ${bold}        | ${True} |

        This keyword works only if lxml mode is taken into use when `importing`
        the library. New in Robot Framework 2.8.5.
        """
        if not self.lxml_etree:
            raise RuntimeError("'Evaluate Xpath' keyword only works in lxml mode.")
        return self.get_element(source, context).xpath(expression)


class NameSpaceStripper(object):

    def strip(self, elem, lwrrent_ns=None):
        if elem.tag.startswith('{') and '}' in elem.tag:
            ns, elem.tag = elem.tag[1:].split('}', 1)
            if ns != lwrrent_ns:
                elem.attrib['xmlns'] = ns
                lwrrent_ns = ns
        elif lwrrent_ns:
            elem.attrib['xmlns'] = ''
            lwrrent_ns = None
        for child in elem:
            self.strip(child, lwrrent_ns)

    def unstrip(self, elem, lwrrent_ns=None):
        ns = elem.attrib.pop('xmlns', lwrrent_ns)
        if ns:
            elem.tag = '{%s}%s' % (ns, elem.tag)
        for child in elem:
            self.unstrip(child, ns)


class ElementFinder(object):

    def __init__(self, etree, modern=True, lxml=False):
        self.etree = etree
        self.modern = modern
        self.lxml = lxml

    def find_all(self, elem, xpath):
        xpath = self._get_xpath(xpath)
        if xpath == '.':  # ET < 1.3 does not support '.' alone.
            return [elem]
        if not self.lxml:
            return elem.findall(xpath)
        finder = self.etree.ETXPath(xpath)
        return finder(elem)

    def _get_xpath(self, xpath):
        if not xpath:
            raise RuntimeError('No xpath given.')
        if self.modern:
            return xpath
        try:
            return str(xpath)
        except UnicodeError:
            if not xpath.replace('/', '').isalnum():
                logger.warn('XPATHs containing non-ASCII characters and '
                            'other than tag names do not always work with '
                            'Python/Jython versions prior to 2.7. Verify '
                            'results manually and consider upgrading to 2.7.')
            return xpath


class ElementComparator(object):

    def __init__(self, comparator, normalizer=None, exclude_children=False):
        self._comparator = comparator
        self._normalizer = normalizer or (lambda text: text)
        self._exclude_children = exclude_children

    def compare(self, actual, expected, location=None):
        if not location:
            location = Location(actual.tag)
        self._compare_tags(actual, expected, location)
        self._compare_attributes(actual, expected, location)
        self._compare_texts(actual, expected, location)
        if location.is_not_root:
            self._compare_tails(actual, expected, location)
        if not self._exclude_children:
            self._compare_children(actual, expected, location)

    def _compare_tags(self, actual, expected, location):
        self._compare(actual.tag, expected.tag, 'Different tag name', location,
                      should_be_equal)

    def _compare(self, actual, expected, message, location, comparator=None):
        if location.is_not_root:
            message = "%s at '%s'" % (message, location.path)
        if not comparator:
            comparator = self._comparator
        comparator(actual, expected, message)

    def _compare_attributes(self, actual, expected, location):
        self._compare(sorted(actual.attrib), sorted(expected.attrib),
                      'Different attribute names', location, should_be_equal)
        for key in actual.attrib:
            self._compare(actual.attrib[key], expected.attrib[key],
                          "Different value for attribute '%s'" % key, location)

    def _compare_texts(self, actual, expected, location):
        self._compare(self._text(actual.text), self._text(expected.text),
                      'Different text', location)

    def _text(self, text):
        return self._normalizer(text or '')

    def _compare_tails(self, actual, expected, location):
        self._compare(self._text(actual.tail), self._text(expected.tail),
                      'Different tail text', location)

    def _compare_children(self, actual, expected, location):
        self._compare(len(actual), len(expected), 'Different number of child elements',
                      location, should_be_equal)
        for act, exp in zip(actual, expected):
            self.compare(act, exp, location.child(act.tag))


class Location(object):

    def __init__(self, path, is_root=True):
        self.path = path
        self.is_not_root = not is_root
        self._children = {}

    def child(self, tag):
        if tag not in self._children:
            self._children[tag] = 1
        else:
            self._children[tag] += 1
            tag += '[%d]' % self._children[tag]
        return Location('%s/%s' % (self.path, tag), is_root=False)
