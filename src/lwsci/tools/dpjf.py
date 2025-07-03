#!/usr/bin/elw python3

import lxml.etree as ET
import sys
import os
import jama_client
import subprocess as sp
import argparse
import re

def findFile(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def findDir(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs:
            return os.path.join(root, name)

def findFilePath(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root

# Function uses XSLT files and transforms xml to xhtml.
def transformXmlToHtml(elementNode,
                       elementType,
                       elementName,
                       implementsJamaId,
                       debug=False):
    # Update file location to remove prefix before "fusa/capture".
    if elementType != "page":
        locationNode = elementNode.xpath("./location")[0]
        locationNodeText = locationNode.attrib["file"]
        locationNodeText = re.sub("(.*)drivers/lwsci", "https://lmi-hub/source/xref/stage-main/vendor/lwpu/cheetah/gpu/drv/drivers/lwsci", locationNodeText)
        locationNode.attrib["file"] = locationNodeText
    # Locate XSLT for xml to html transformation
    pathToXsltXml = findFile("{}XmlToXhtml.xslt".format(elementType), ".")
    xsltXML = ET.parse(pathToXsltXml)
    xsltTransformObj = ET.XSLT(xsltXML)
    result = xsltTransformObj(elementNode)
    if debug:
        print("Found XSLT file at:{}".format(pathToXsltXml))
        outputHtmlDir = "outputHtml"
        if (findDir(outputHtmlDir, ".") is None):
            print("{0} not found. Creating {0}".format(outputHtmlDir))
            os.mkdir(outputHtmlDir, 0o775)

        pathForHtmlOutput = "./{}/{}{}.html".format(outputHtmlDir, elementType, elementName)
        print("Dumping Html output at:{}".format(pathForHtmlOutput))
        result.write(pathForHtmlOutput)
    print("\t{}Ancestor:{}<JamaId:{}>'s xml has been colwerted to xhtml".format(elementType, elementName, implementsJamaId))
    return str(result)

# Function takes html output and pushes it to Jama SWUD atoms.
def populateJamaSWUDAtom(jc, implementsJamaId, atomName, doxygenHtml):
    unitDesignAtom = jc.get_by_id(implementsJamaId)
    if unitDesignAtom is None:
        return
    if unitDesignAtom.name == atomName and unitDesignAtom.description == doxygenHtml:
        print("JAMA item {} identical to exported text, skipping...\n".format(implementsJamaId))
        return
    print(unitDesignAtom.description)
    print(unitDesignAtom.name)
    unitDesignAtom.set_attr("name", atomName)
    print(unitDesignAtom.name)
    unitDesignAtom.set_attr("description", doxygenHtml)
    try:
        unitDesignAtom.save()
        print("\tUnitDesignAtom with JamaId:{} Name:{} updated".format(implementsJamaId, atomName))
    except:
        print("Failed to update item " + implementsJamaId + " referring to name " + atomName)

# Function to handle ancestor elements for each @implements tag in header files.
def handleAncestor(ancestorList,
                   jc,
                   implementsJamaId,
                   debug):
    for ancestor in ancestorList:
        ancestorKind = ancestor.attrib["kind"]
        if ((ancestorKind == "struct") or (ancestorKind == "class")):
            ancestorName = ancestor.find("compoundname").text
        elif (ancestorKind == "page"):
            ancestorName = ancestor.find("title").text
        else:
            ancestorName = ancestor.find("name").text
        doxygenHtml = transformXmlToHtml(ancestor, ancestorKind, ancestorName, implementsJamaId, debug)
        #added support to export the fragment name with namespace for c++ functions
        if (ancestorKind == "function"):
            #confirm whether it is C++ function
            ancestorNameWithNameSpace = ancestor.find("definition").text
            namespace_found = ancestorNameWithNameSpace.find("::")
            if (namespace_found != -1):
                #for static/virtual function, we should omit the keyword static/virtual
                virtual_index = ancestorNameWithNameSpace.find("virtual")
                static_index = ancestorNameWithNameSpace.find("static")
                if (((virtual_index != -1) and (virtual_index == 0)) or ((static_index != -1) and (static_index == 0))):
                    space_index = ancestorNameWithNameSpace.find(" ")
                    if space_index != -1:
                        ancestorNameWithNameSpace = ancestorNameWithNameSpace[space_index + 1:]
                #remove the return type from definition if any
                if (ancestor.find("type").text or ancestor.findtext("./type/ref")):
                    namespace_index = ancestorNameWithNameSpace.find(" ")
                    if namespace_index != -1:
                        ancestorNameWithNameSpace = ancestorNameWithNameSpace[namespace_index + 1:]
                populateJamaSWUDAtom(jc, implementsJamaId, "{} {}".format(ancestorKind, ancestorNameWithNameSpace), doxygenHtml)
            else:
                #This fix was added to removed :: stray char in functions. Tested without this fix and API functions
                #are updating properly without :: character. So this fix is not needed anymore.
                #doxygenHtml = doxygenHtml.replace('::', '')
                populateJamaSWUDAtom(jc, implementsJamaId, "{} {}".format(ancestorKind, ancestorName), doxygenHtml)
        elif (ancestorKind == "typedef"):
            definition = ancestor.find("definition").text
            if (definition.startswith("using")):
                populateJamaSWUDAtom(jc, implementsJamaId, "using {}".format(ancestorName), doxygenHtml)
            else:
                populateJamaSWUDAtom(jc, implementsJamaId, "{} {}".format(ancestorKind, ancestorName), doxygenHtml)
        else:
            populateJamaSWUDAtom(jc, implementsJamaId, "{} {}".format(ancestorKind, ancestorName), doxygenHtml)

# main function
def main(argv):
    # Instantiate parser arguments
    parser = argparse.ArgumentParser(description='Doxygen Population to Jama Framework')
    parser.add_argument("projectID",       action="store", help="the Jama ID of the project containing the SWUD")
    parser.add_argument("doxyOutput",      action="store", help="the folder containing doxygen output")
    parser.add_argument("--clientID",      action="store",
                        help="the client ID of the user. Can be provided via CLIENT_ID environment variable as well.",
                        metavar="CID")
    parser.add_argument("--clientSecret",  action="store",
                        help="the client secret of the user. Can be provided via CLIENT_SECRET environment variable as well.",
                        metavar="CST")
    parser.add_argument("-d", "--debug",   action="store_true", dest="debug", help="turns on debug html output(default=false)")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")

    # Parse arguments
    args = parser.parse_args()

    # Check if clientID is none and try to pull from environment variable.
    if (args.clientID is None):
        print("No client ID provided as argument.")
        try:
            args.clientID = os.elwiron["CLIENT_ID"]
        except KeyError:
            print("Expected CLIENT_ID to be defined as environment variable if not provided as an argument.")
            print("Set and try again.")
            exit(1)

    # Check if clientSecret is none and try to pull from environment variable.
    if (args.clientSecret is None):
        print("No client secret provided")
        try:
            args.clientSecret = os.elwiron["CLIENT_SECRET"]
        except KeyError:
            print("Expected CLIENT_SECRET to be defined as environment variable if not provided as an argument.")
            print("Set and try again.")
            exit(1)

    # Instantiate jama client object
    jc = jama_client.JamaClient("https://lwpu.jamacloud.com/rest", args.projectID, args.clientID, args.clientSecret)

    # TODO: Check for exceptions while instantiating jama client.

    # Doxygen provides a combine.xslt file by default that helps
    # combine all xml output files into one gigantic xml file using
    # the ref-ids provided in index.xml file.
    # index.xml file is also produced by default with doxygen xml output.
    pathToCombineXslt = findFilePath("combine.xslt", args.doxyOutput)
    if args.debug:
        print("Found combine.xslt at:{}".format(pathToCombineXslt))

    # Prepare command string for all xml combination.
    commandString = "cd {} && xsltproc combine.xslt index.xml >all.xml".format(pathToCombineXslt)
    if args.debug:
        print("Going to execute command:\"{}\"".format(commandString))
    result = sp.run(commandString, shell=True)
    # TODO: Check for result and catch exceptions.

    # Find the newly combined all.xml
    pathToInputXml = findFile("all.xml", args.doxyOutput)
    if args.debug:
        print("Found allxml at:{}".format(pathToInputXml))

    # Parse all.xml and extract root node.
    tree = ET.parse(pathToInputXml)
    root = tree.getroot()

    # Find all jama-implements tags.
    for element in root.xpath(".//detaileddescription/para/jama-implements"):
        # Save Jama Id of unit design atom
        implementsJamaId = element.text
        fillableSet = set(["11566706"])
#uncomment below to filter by jama unique id
#        if implementsJamaId not in fillableSet:
#            continue
#uncomment below to skip those entries
        skipSet = set(["11409713", "11410103", "11410064"])
#        if implementsJamaId in skipSet:
#            continue
        # Try searching for supported types of ancestors
        # List of supported types includes:
        # 1. Define
        # 2. Enum
        # 3. Function
        # 4. Struct
        # 5. Class
        # Search style 1 using xpath
        defineAncestorList   = element.xpath("ancestor::memberdef[@kind='define']")
        enumAncestorList     = element.xpath("ancestor::memberdef[@kind='enum']")
        functionAncestorList = element.xpath("ancestor::memberdef[@kind='function']")
        structAncestorList   = element.xpath("ancestor::compounddef[@kind='struct']")
        classAncestorList    = element.xpath("ancestor::compounddef[@kind='class']")
        typedefAncestorList  = element.xpath("ancestor::memberdef[@kind='typedef']")
        variableAncestorList = element.xpath("ancestor::memberdef[@kind='variable']")
        if (defineAncestorList):
            handleAncestor(defineAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (enumAncestorList):
            handleAncestor(enumAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (functionAncestorList):
            handleAncestor(functionAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (structAncestorList):
            handleAncestor(structAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (classAncestorList):
            handleAncestor(classAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (typedefAncestorList):
            handleAncestor(typedefAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)
        elif (variableAncestorList):
            handleAncestor(variableAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)

    # Find all jama-implements tags.
    for element in root.xpath(".//detaileddescription/sect1/para/jama-implements"):
        # Save Jama Id of unit design atom
        implementsJamaId = element.text
        blanketStatementAncestorList = element.xpath("ancestor::compounddef[@kind='page']")
        if (blanketStatementAncestorList):
            handleAncestor(blanketStatementAncestorList,
                           jc,
                           implementsJamaId,
                           args.debug)


if __name__ == "__main__":
    main(sys.argv)
