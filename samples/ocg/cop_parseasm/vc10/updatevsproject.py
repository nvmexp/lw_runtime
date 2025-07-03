#!/usr/bin/elw python
"""
genProjectFile.py : Generate project file by

looking at: *.obj_cl_params files

   Below 2 preprocessing commands are needed:
   $ find $PWD/_out/wddm_x86_debug/ -name "*.obj_cl_params" >> files.txt
   # replace wddm2_amd64_debug by whatever build variant you use
   $ cat `cat files.txt` > dump.log

usage:
       $ python genProjectFile.py dump.log
dependencies:
       needs lxml for xml manipulations
       $ pip install lxml # needs cygwin python
Assumptions: script assumes $P4ROOT is on c:/src
This script is still incipient please contact mvaidya@lwpu.com for updating
the spirvc project.

If you want to add new build configuration, you need to change projecttemplate.xml
"""

import sys
import re
import uuid
from lxml import etree as ET
import itertools
import os

"""
make categories to add filters
category in beginning takes precedence
e.g.
categories = [
    (re.compile(r".*/ori/.*fermi.*"), "cop_ori_fermi"),
    (re.compile(r".*/ori/.*volta.*"), "cop_ori_volta"),
    ]

"""
categories = []

"""
Some const definitions we will be using through out
we will be using this namespace in xml
"""
ns = {"ns" : "http://schemas.microsoft.com/developer/msbuild/2003"}

srcLineRegex = re.compile(r"[Cc]:")
includeFilesRegex = re.compile(r"^-I")
macroRegex = re.compile(r"^-D")
p4rootRegex = re.compile(r"c:/src", re.IGNORECASE)
projectName = "parseasm"
RawLog = False

if len(sys.argv) > 2:
    assert sys.argv[2] == "rawlog"
    RawLog = True

compilerParamsRegex = re.compile(r"^compiler params .*") if RawLog else re.compile(r".*")

excludeRegex = re.compile(r".*[.]obj_cl_params[:].*")

driverRegex = re.compile(r".*/sw/dev/gpu_drv/module_compiler/(.*)")

toolsRegex = re.compile(r".*/sw/tools/(.*)")
generatedIncludes = re.compile(r"^_out/.*")

"""
Get only those lines which are compiler params
"""
lines = " ".join(filter(lambda x: compilerParamsRegex.match(x), ("".join(open(sys.argv[1]).readlines()).replace('\r', '\n').split('\n'))))

lines = lines.replace('\n', ' ').replace('\r', ' ').replace('\\', '/').replace('//', '/').split()
lines = set(lines)

lines = set(filter(lambda x: not excludeRegex.match(x), lines))

filterTree = ET.Element("Project", {"ToolsVersion" : "4.0", "xmlns" : "http://schemas.microsoft.com/developer/msbuild/2003"})

"""
for project files, load template and update interesting values
"""

projTree = ET.parse('projecttemplate.xml', ET.XMLParser(remove_blank_text=True)).getroot()

filterList = ET.SubElement(filterTree, "ItemGroup")

"""
get macros aka -DFlags and include paths aka -Ipaths
"""
macros = list(set(
    (map(lambda x: x.strip(',"\''),
         filter(lambda x: macroRegex.match(x), lines)))))
includes = list(set(
    (map(lambda x: x.strip(',"\''),
         filter(lambda x: includeFilesRegex.match(x), lines)))))
"""
remove earlier tags and clear empty itemgroups
"""
for ClCompile in projTree.xpath('//ns:ClCompile', namespaces=ns):
    ClCompile.getparent().remove(ClCompile)
for ClCompile in projTree.xpath('//ns:ClInclude', namespaces=ns):
    ClCompile.getparent().remove(ClCompile)

for itemGroup in projTree.xpath('ns:ItemGroup', namespaces=ns):
    if len(itemGroup.getchildren()) == 0:
        itemGroup.getparent().remove(itemGroup)

lines = set(filter(lambda x: srcLineRegex.match(x),
                   (map(lambda x: x.strip(',"\''), lines))))

"""
Build categories using pathnames
replace spaces by _
"""
dirs = map(lambda x: os.path.dirname(x),
           filter(lambda x: len(x) != 0,
              map((lambda x:
                   driverRegex.match(x).group(1) if driverRegex.match(x) else ""), lines)))

"""
Get canonical closure of dirs
"""
closure = set()
for directory in dirs:
    fromRoot = ""
    for component in directory.split('/'):
        if len(fromRoot):
            fromRoot = fromRoot + "/"
        fromRoot += component
        closure.add(fromRoot)

sortedClosure = sorted(closure, key=lambda file: (os.path.dirname(file), os.path.basename(file)), reverse = True)

for directory in sortedClosure:
    categories.append((re.compile(r".*" + directory + ".*"), re.sub(r'\s', '_', directory).replace('/', '\\')))

for category in categories:
    categoryName = category[1]
    ET.SubElement(ET.SubElement(filterList, "Filter", {"Include" : categoryName}),
                  "UniqueIdentifier").text = "{" + str(uuid.uuid4()) + "}"

for category in categories:
    # ET.SubElement(ET.SubElement(filterList, "Filter", {"Include" : category[1]}),
    #               "UniqueIdentifier").text = "{" + str(uuid.uuid4()) + "}"

    filterItemGroup = ET.SubElement(filterTree, "ItemGroup")
    projItemGroup = ET.SubElement(projTree, "ItemGroup")

    categoryRegex = category[0]
    sourceLines = set(filter(lambda x: categoryRegex.match(x), lines))
    lines = set(lines) - set(sourceLines)
    # sourceLines.sort()
    for line in sorted(sourceLines):
        # assert driverRegex.match(line)
        line = "../../../../" + driverRegex.match(line).group(1)
        ET.SubElement(ET.SubElement(filterItemGroup, "ClCompile", {"Include" : line}), "Filter").text = category[1]
        ET.SubElement(projItemGroup, "ClCompile", {"Include" : line})


for NMakeIncludeSearchPath in projTree.xpath('//ns:NMakeIncludeSearchPath', namespaces=ns):
    NMakeIncludeSearchPath.text = ';'.join(
        map(lambda x: ("../" + x if generatedIncludes.match(x) else x),
            map(lambda x: (("$(LW_TOOLS)/" + toolsRegex.match(x).group(1)) if toolsRegex.match(x) else x),
                map(lambda x: (("../../../../" + driverRegex.match(x[2:]).group(1)) if (driverRegex.match(x) and True) else x[2:]), includes))))

for NMakePreprocessorDefinitions in projTree.xpath('//ns:NMakePreprocessorDefinitions', namespaces=ns):
    NMakePreprocessorDefinitions.text = ';'.join(map(lambda x: x[2:], macros))


ET.ElementTree(filterTree).write(projectName + ".vcxproj.filters", encoding='utf-8', xml_declaration = True, pretty_print = True)
ET.ElementTree(projTree).write(projectName + ".vcxproj", encoding='utf-8', xml_declaration = True, pretty_print = True)
