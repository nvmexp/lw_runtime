import os
import os.path
import sys
import argparse
import logging
import datetime
import fileinput

# choosing a global name allows to share this instance between different parts of the app
log = logging.getLogger(__name__)

def extractExtensionName(filePath):
    (name, dummy) = os.path.splitext(os.path.basename(filePath))
    if len(name) < 4 or not name.startswith('Ext'):
        raise Exception(f'Given name {name} is not a valid extension name. Extension names start with \'Ext\'.')
    return name[3:]

def getNextValidExtensionABIVersion(lastKnownABI):
    # lwrrently intended step width is 1000 to allow for different versions
    return int(lastKnownABI)+1000

def extractExtABIVersionFromComment(line):
    # return either the defined extension ABI version or 0 else
    # line must be of this special format
    # // <value> - <extension ABI version name>
    if line.startswith('// ') and line.find('OPTIX_EXT_') != -1:
        tokens = line.split()
        if len(tokens) >= 4 and tokens[3].startswith('OPTIX_EXT_'):
            if tokens[1].isdigit():
                return int(tokens[1])
    return 0

# Retrieve the last listed extension ABI version - which is dolwmented inside the comments right above optixQueryFunctionTable().
# It is actually the very last of such comment lines of the same format
# // <value> - <extension ABI version name>
def getLastKnownExtABI(file):
    inABIDefines = False
    lastKnownExtABI = None
    with open(file) as f:
        for line in f.readlines():
            # scan all existing ABI versions and increase last one by one
            if line.startswith('// ') and line.find('OPTIX_EXT_') != -1:
                tokens = line.split()
                if len(tokens) >= 4 and tokens[3].startswith('OPTIX_EXT_'):
                    inABIDefines = True
                    if lastKnownExtABI and int(lastKnownExtABI) >= int(tokens[1]):
                        raise Exception(f'list of OPTIX_EXT_XXX_ABI_VERSION must be sorted')
                    lastKnownExtABI = tokens[1]
                else:
                    inABIDefines = False
            else:
                inABIDefines = False
            if lastKnownExtABI and not inABIDefines:
                return lastKnownExtABI
    raise Exception(f'Failed to find extension ABI version in file {file}')

class PublicExtHeaderGenerator:
    def __init__(self, fileName, extensionName, extensionABIVersion):
        self._fileName = os.path.basename(fileName)
        (self._name, dummy) = os.path.splitext(self._fileName)
        self._year = datetime.date.today().year
        self._extensionName = extensionName
        self._extensionABIVersion = extensionABIVersion

    def headerText(self):
        line_sep = '\n'
        return line_sep.join(self._header_lines()) + line_sep

    def introText(self):
        line_sep = '\n'
        return line_sep.join(self._intro_lines()) + line_sep

    def theExtHeaderText(self):
        line_sep = '\n'
        return line_sep.join(self._theExtHeader_lines()) + line_sep

    def theExtStubsHeaderText(self):
        line_sep = '\n'
        return line_sep.join(self._theExtStubsHeader_lines()) + line_sep

    def theExtFunctionTableDefText(self):
        line_sep = '\n'
        return line_sep.join(self._theExtFunctionTableDef_lines()) + line_sep

    def implText(self):
        line_sep = '\n'
        return line_sep.join(self._implText_lines()) + line_sep

    def functionTableImpl(self):
        line_sep = '\n'
        return line_sep.join(self._functionTableImpl_lines()) + line_sep

    def _implText_lines(self):
        yield '#include <optix_ext_{}.h>'.format(self._extensionName.lower())
        yield ''
        yield 'static_assert(false, "Add implementation of functions here - as extern \\"C\\" functions, eg\\n\\\n  extern \\"C\\" OptixResult optixExt{}FunctionName( Foo* bar ) /*...*/;");'.format(self._extensionName)
        yield ''

    def _header_lines(self):
        yield '/*'
        yield ' * Copyright (c) {} LWPU Corporation.  All rights reserved.'.format(self._year)
        yield ' *'
        yield ' * LWPU Corporation and its licensors retain all intellectual property and proprietary'
        yield ' * rights in and to this software, related documentation and any modifications thereto.'
        yield ' * Any use, reproduction, disclosure or distribution of this software and related'
        yield ' * documentation without an express license agreement from LWPU Corporation is strictly'
        yield ' * prohibited.'
        yield ' *'
        yield ' * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*'
        yield ' * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,'
        yield ' * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A'
        yield ' * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY'
        yield ' * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT'
        yield ' * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF'
        yield ' * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR'
        yield ' * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF'
        yield ' * SUCH DAMAGES'
        yield ' */'
        yield ''

    def _intro_lines(self):
        yield '/**'
        yield ' * @file   {}'.format(self._fileName)
        yield ' * @author LWPU Corporation'
        yield ' * @brief  OptiX public API header'
        yield ' *'
        yield ' */'
        yield ''
        yield '#ifndef __{}_h__'.format(self._name)
        yield '#define __{}_h__'.format(self._name)
        yield ''

    def _theExtHeader_lines(self):
        yield '#include "optix_types.h"'
        yield ''
        yield '#ifdef __cplusplus'
        yield 'extern "C" {'
        yield '#endif'
        yield ''
        yield 'static_assert(false, "Add the extension here, eg\\n\\\n  OptixResult optixExt{}FunctionName( Foo* bar );");'.format(self._extensionName)
        yield ''
        yield '#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION'
        yield '// When changing the ABI version make sure you know exactly what you are doing. See'
        yield '// apps/optix/exp/functionTable/functionTable.cpp for instructions. See'
        yield '// https://confluence.lwpu.com/display/RAV/ABI+Versions+in+the+Wild for released ABI versions.'
        yield '#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION'
        yield '#define OPTIX_EXT_{}_ABI_VERSION {}'.format(self._extensionName.upper(), self._extensionABIVersion)
        yield ''
        yield 'typedef struct OptixExt{}FunctionTable'.format(self._extensionName)
        yield '{'
        yield 'static_assert(false, "Define corresponding function pointer here, eg\\n\\\n  OptixResult ( *optixExt{}FunctionName )( Foo* bar );");'.format(self._extensionName)
        yield '}'
        yield '   OptixExt{}FunctionTable;'.format(self._extensionName)
        yield ''
        yield '#ifdef __cplusplus'
        yield '}'
        yield '#endif'
        yield ''
        yield '#endif /* __{}_h__ */'.format(self._name.lower())

    def _theExtStubsHeader_lines(self):
        yield '#include "optix_ext_{}.h"'.format(self._extensionName)
        yield ''
        yield '#ifdef _WIN32'
        yield '#ifndef WIN32_LEAN_AND_MEAN'
        yield '#define WIN32_LEAN_AND_MEAN 1'
        yield '#endif'
        yield '#include <windows.h>'
        yield '#else'
        yield '#include <dlfcn.h>'
        yield '#endif'
        yield ''
        yield '#ifdef __cplusplus'
        yield 'extern "C" {'
        yield '#endif'
        yield ''
        yield '/* The function table needs to be defined in exactly one translation unit. This can be'
        yield '   achieved by including optix_ext_{}_function_table_definition.h in that translation unit.'.format(self._extensionName)
        yield ' */'
        yield 'extern OptixExt{}FunctionTable g_optixExt{}FunctionTable;'.format(self._extensionName, self._extensionName)
        yield ''
        yield '// Initializes the function table used by the stubs for the extension API for compiler_version.'
        yield '//'
        yield '// The function requires a handle to the loaded OptiX library. This handle can be obtained by using'
        yield '// optixInitWithHandle() instead of optixInit(), for example (error handling ommitted):'
        yield '//'
        yield '//     void* handle;'
        yield '//     optixInitWithHandle( &handle );'
        yield '//     optixExt{}Init( handle );'.format(self._extensionName)
        yield '//'
        yield 'inline OptixResult optixExt{}Init( void* handle )'.format(self._extensionName)
        yield '{'
        yield '    if( !handle )'
        yield '        return OPTIX_ERROR_ILWALID_VALUE;'
        yield ''
        yield '#ifdef _WIN32'
        yield '    void* symbol = GetProcAddress( (HMODULE)handle, "optixQueryFunctionTable" );'
        yield '    if( !symbol )'
        yield '        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;'
        yield '#else'
        yield '    void* symbol = dlsym( handle, "optixQueryFunctionTable" );'
        yield '    if( !symbol )'
        yield '        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;'
        yield '#endif'
        yield ''
        yield '    OptixQueryFunctionTable_t* optixQueryFunctionTable = (OptixQueryFunctionTable_t*)symbol;'
        yield ''
        yield '    return optixQueryFunctionTable( OPTIX_EXT_{}_ABI_VERSION, 0, 0, 0, &g_optixExt{}FunctionTable,'.format(self._extensionName.upper(), self._extensionName)
        yield '                                    sizeof( g_optixExt{}FunctionTable ) );'.format(self._extensionName)
        yield '}'
        yield ''
        yield '/* Stub functions that forward calls to the corresponding function pointer in the function table. */'
        yield 'static_assert(false, "Add stub definition here, eg\\n\\\n  inline OptixResult optixExt{}FunctionName( Foo* bar ) ...");'.format(self._extensionName)
        yield ''
        yield '#ifdef __cplusplus'
        yield '}'
        yield '#endif'
        yield ''
        yield '#endif /* __{}_h__ */'.format(self._name.lower())

    def _theExtFunctionTableDef_lines(self):
        yield '#include "optix_ext_{}.h"'.format(self._extensionName)
        yield ''
        yield '#ifdef __cplusplus'
        yield 'extern "C" {'
        yield '#endif'
        yield ''
        yield '/* If the stubs in optix_ext_{}_stubs.h are used, then the function table needs to be'.format(self._extensionName.lower())
        yield '   defined in exactly one translation unit. This can be achieved by including this header file in'
        yield '   that translation unit. */'
        yield 'OptixExt{}FunctionTable g_optixExt{}FunctionTable;'.format(self._extensionName, self._extensionName)
        yield ''
        yield '#ifdef __cplusplus'
        yield '}'
        yield '#endif'
        yield ''
        yield '#endif /* __{}_h__ */'.format(self._name.lower())

    def _functionTableImpl_lines(self):
        yield '#define OPTIX_OPTIONAL_FEATURE_OPTIX7'
        yield '#include <optix_ext_{}.h>'.format(self._extensionName.lower())
        yield '#include <optix_types.h>'
        yield ''
        yield '#include <cstring>'
        yield ''
        yield 'namespace optix_exp {'
        yield ''
        yield 'namespace {'
        yield ''
        yield '// This struct is just a permanent copy of the then-current struct OptixExt{}FunctionTable in'.format(self._extensionName)
        yield '// optix_ext_{}.h.'.format(self._extensionName.lower())
        yield '//'
        yield '// We could use an array of void* here, but the explicit types prevent mistakes like ordering'
        yield '// problems or signature changes of functions used in tables of released ABI versions.'
        yield 'struct FunctionTableExt{}_lwrrent'.format(self._extensionName)
        yield '{'
        yield 'static_assert(false, "Add your extension function pointer definitions here, eg\\n\\\n  OptixResult ( *optixExt{}FunctionName )( Foo* bar );");'.format(self._extensionName)
        yield '};'
        yield ''
        yield 'static_assert(false, "Add your extension function pointer to the following table, eg\\n\\\n  optixExt{}FunctionName");'.format(self._extensionName)
        yield 'FunctionTableExt{}_lwrrent g_functionTableExt{}_lwrrent = {{'.format(self._extensionName, self._extensionName)
        yield '    // clang-format off'
        yield '    // clang-format on'
        yield '};'
        yield '}'
        yield ''
        yield 'OptixResult fillFunctionTableExt{}_lwrrent( void* functionTable, size_t sizeOfFunctionTable )'.format(self._extensionName)
        yield '{'
        yield '    if( sizeOfFunctionTable != sizeof( FunctionTableExt{}_lwrrent ) )'.format(self._extensionName)
        yield '        return OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH;'
        yield ''
        yield '    memcpy( functionTable, &g_functionTableExt{}_lwrrent, sizeof( FunctionTableExt{}_lwrrent ) );'.format(self._extensionName, self._extensionName)
        yield '    return OPTIX_SUCCESS;'
        yield '};'
        yield '}'

class ExtensionGenerator:
    def __init__(self, extensionImplFilePath):
        self._extensionImplFilePath = extensionImplFilePath
        self._extensionName = extractExtensionName(extensionImplFilePath)
        log.info(f'handling extension {self._extensionName}')
        self._ABIversion = None
        self._localLwmakeFile = None
        self._rootDir = None

        # some correctness checks
        self._checkCorrectDirectoryLayout()
        self._checkWritability()

    def generateFiles(self):
        log.info(f'generate files for extension {self._extensionName}')
        self._getLwrrentABIVersion()
        self._generateExtensionHeaders()
        self._generateImplFile()
        self._generateFunctionTables()

    def _generateImplFile(self):
        file = self._extensionImplFilePath
        log.info(f'generating implementation {file}')
        generator = PublicExtHeaderGenerator(file, self._extensionName, self._ABIversion)
        with open(file, 'w') as f:
            f.write(generator.headerText())
            f.write(generator.implText())
        # add file to corresponding .lwmk file
        if self._localLwmakeFile is None:
            raise Exception(f'local .lwmk file was not found in directory {os.path.dirname(file)}')
        if os.path.dirname(file) != os.path.dirname(self._localLwmakeFile):
            raise Exception(f'imple file {file} and local .lwmk file {self._localLwmakeFile} seem to be in different directories')
        log.warning(f'Please add the new file {os.path.basename(file)} to the .lwmk file {self._localLwmakeFile}')

    def _generateExtensionHeaders(self):
        if self._rootDir is None:
            raise Exception(f'OptiX root directory is not set')
        file = 'optix_ext_' + self._extensionName.lower() + '.h'
        file = os.path.join(self._rootDir, 'include', file)
        self._generateExtensionHeader(file)
        file = os.path.basename(file)
        (file, ext) = os.path.splitext(file)
        postfix = '_stubs'
        file += postfix
        file += ext
        file = os.path.join(self._rootDir, 'include', file)
        self._generateExtensionStubs(file)
        log.warning(f'Please don\'t forget to edit the new stubs header file {file}')
        file = os.path.basename(file)
        (file, ext) = os.path.splitext(file)
        file = file[:-len(postfix)]
        file += '_function_table_definition'
        file += ext
        file = os.path.join(self._rootDir, 'include', file)
        self._generateExtensionFunctionTableDef(file)

    def _generateExtensionHeader(self, file):
        log.info(f'generating extension header {file}')
        generator = PublicExtHeaderGenerator(file, self._extensionName, self._ABIversion)
        with open(file, 'w') as f:
            f.write(generator.headerText())
            f.write(generator.introText())
            f.write(generator.theExtHeaderText())

    def _generateExtensionStubs(self, file):
        log.info(f'generating extension stubs header {file}')
        generator = PublicExtHeaderGenerator(file, self._extensionName, self._ABIversion)
        with open(file, 'w') as f:
            f.write(generator.headerText())
            f.write(generator.introText())
            f.write(generator.theExtStubsHeaderText())

    def _generateExtensionFunctionTableDef(self, file):
        log.info(f'generating extension function table definition {file}')
        generator = PublicExtHeaderGenerator(file, self._extensionName, self._ABIversion)
        with open(file, 'w') as f:
            f.write(generator.headerText())
            f.write(generator.introText())
            f.write(generator.theExtFunctionTableDefText())

    def _checkCorrectDirectoryLayout(self):
        if os.path.exists(self._extensionImplFilePath):
            raise Exception(f'file {self._extensionImplFilePath} does already exist. Exiting...')
        if not os.path.exists(os.path.dirname(self._extensionImplFilePath)):
            raise Exception(f'directory {os.path.dirname(self._extensionImplFilePath)} does not exist')

        directories = []
        (head, tail) = os.path.split(os.path.dirname(self._extensionImplFilePath))
        while head and tail:
            directories.append(tail)
            (head, tail) = os.path.split(head)
        if head:
            directories.append(head)
        # ensure layout apps\optix\exp\Foo
        if len(directories) < 5:
            raise Exception(f'directory {os.path.dirname(self._extensionImplFilePath)} cannot be part of full OptiX 7 code base')
        if directories[1] != 'exp' or directories[2] != 'optix' or directories[3] != 'apps':
            raise Exception(f'directory {os.path.dirname(self._extensionImplFilePath)} cannot be part of full OptiX 7 code base')
        # saving the root directory
        for d in reversed(directories):
            self._rootDir = os.path.join(str(self._rootDir), d)
            if d == 'optix':
                break
        log.info(f'RootDirectory is {self._rootDir}')

    def _checkWritability(self):
        if self._rootDir is None:
            raise Exception(f'OptiX root directory is not set')
        # generate list to make it easier for user to get overview of required files
        requiredReadableFiles = []
        # check corresponding .lwmk File
        for f in os.listdir(os.path.dirname(self._extensionImplFilePath)):
            (root, ext) = os.path.splitext(f)
            if ext != '.lwmk':
                continue
            file = os.path.join(os.path.dirname(str(self._extensionImplFilePath)), str(f))
            self._localLwmakeFile = file
            requiredReadableFiles.append(file)
            break
        # check functionTable files
        file = os.path.join(self._rootDir, 'exp', 'functionTable', 'functionTable.cpp')
        requiredReadableFiles.append(file)
        file = os.path.join(self._rootDir, 'exp', 'functionTable', 'functionTable.lwmk')
        requiredReadableFiles.append(file)

        for file in requiredReadableFiles:
            if not os.access(file, os.W_OK):
                raise Exception(f'required file {file} not readable. Please ensure that all of these files are readable/checked out\n{requiredReadableFiles}')

    def _getLwrrentABIVersion(self):
        if self._rootDir is None:
            raise Exception(f'OptiX root directory is not set')
        file = os.path.join(self._rootDir, 'exp', 'functionTable', 'functionTable.cpp')
        if not os.path.exists(file):
            raise Exception(f'Cannot locate file {file}')
        lastKnownABI = getLastKnownExtABI(file)
        self._ABIversion = getNextValidExtensionABIVersion(lastKnownABI)

    def _generateFunctionTables(self):
        if self._rootDir is None:
            raise Exception(f'OptiX root directory is not set')
        # generate specific function table impl
        implFileName = 'functionTableExt' + self._extensionName + '_lwrrent.cpp'
        file = os.path.join(self._rootDir, 'exp', 'functionTable', implFileName)
        self._generateFunctionTableImpl(file)
        # adapt accordingly the functionTable
        file = os.path.join(self._rootDir, 'exp', 'functionTable', 'functionTable.cpp')
        self._addNewFunctionTable(file)
        # and finally add function table impl to .lwmk file
        file = os.path.join(self._rootDir, 'exp', 'functionTable', 'functionTable.lwmk')
        self._addNewFunctionTableToLwmk(file, implFileName)


    def _generateFunctionTableImpl(self, file):
        log.info(f'generating function table implementation {file}')
        generator = PublicExtHeaderGenerator(file, self._extensionName, self._ABIversion)
        with open(file, 'w') as f:
            f.write(generator.headerText())
            f.write(generator.functionTableImpl())

    def _addNewFunctionTable(self, file):
        # for inserting correct #include
        inExtInclusionBlock = False
        inExtInclusionBlockWritten = False
        # for inserting correct function decl
        inFillFunctionDeclBlock = False
        inFillFunctionDeclBlockWritten = False
        # for inserting correct function call
        inFunctionTableCall = False
        inFunctionTableCallWritten = False
        # for inserting new extension ABI version as a comment
        inExtABIVersionsComment = False
        inExtABIVersionsCommentWritten = False
        lastABIVersion = 0
        for line in fileinput.FileInput(file, inplace=True):
            if line.startswith('#include <optix_ext_'):
                if inExtInclusionBlockWritten:
                    raise Exception(f'unexpected layout of optix_ext_XXX.h inclusions')
                inExtInclusionBlock = True
            # add new insert
            elif inExtInclusionBlock and not line.startswith('#include <optix_ext_'):
                print(f'#include <optix_ext_{self._extensionName.lower()}.h>')
                inExtInclusionBlock = False
                inExtInclusionBlockWritten = True
            elif line.startswith('OptixResult fillFunctionTableExt'):
                inFillFunctionDeclBlock = True
            # add function decl
            elif line.startswith('}  // namespace optix_exp'):
                print(f'OptixResult fillFunctionTableExt{self._extensionName}_lwrrent( void* functionTable, size_t sizeOfFunctionTable );')
                print(f'')
                inFillFunctionDeclBlock = False
                inFillFunctionDeclBlockWritten = True
            elif 'return optix_exp::fillFunctionTableExt' in line:
                inFunctionTableCall = True
            elif inFunctionTableCall and 'return OPTIX_ERROR_UNSUPPORTED_ABI_VERSION;' in line:
                print(f'    if( abiId == OPTIX_EXT_{self._extensionName.upper()}_ABI_VERSION )')
                print(f'        return optix_exp::fillFunctionTableExt{self._extensionName}_lwrrent( functionTable, sizeOfFunctionTable );')
                print(f'')
                inFunctionTableCall = False
                inFunctionTableCallWritten = True
            elif extractExtABIVersionFromComment(line):
                inExtABIVersionsComment = True
                lwrrentABIVersion = extractExtABIVersionFromComment(line)
                if lastABIVersion > lwrrentABIVersion:
                    raise Exception(f'Extension ABI versions in optixQueryFunctionTable()\'s comment not sorted')
                lastABIVersion = lwrrentABIVersion
            elif inExtABIVersionsComment:
                if inExtABIVersionsCommentWritten:
                    raise Exception(f'Unexpected layout of extension ABI versions in comment section of optixQueryFunctionTable(), functionTable.cpp')
                print(f'// {self._ABIversion} - OPTIX_EXT_{self._extensionName.upper()}_ABI_VERSION')
                inExtABIVersionsComment = False
                inExtABIVersionsCommentWritten = True
            print(line, end='')
        if inExtInclusionBlock or not inExtInclusionBlockWritten:
            raise Exception(f'Failure in inserting correct #include <> into functionTable.cpp')
        if inFillFunctionDeclBlock or not inFillFunctionDeclBlockWritten:
            raise Exception(f'Failure in inserting correct function decl into functionTable.cpp')
        if inFunctionTableCall or not inFunctionTableCallWritten:
            raise Exception(f'Failure in inserting correct function call into functionTable.cpp')
        if inExtABIVersionsComment or not inExtABIVersionsCommentWritten:
            raise Exception(f'Failure in inserting new extension ABI version into comment section of optixQueryFunctionTable(), functionTable.cpp')

    def _addNewFunctionTableToLwmk(self, file, implFileName):
        inFunctionTableAdds = False
        inFunctionTableAddsWritten = False
        for line in fileinput.FileInput(file, inplace=True):
            if 'functionTableExt' in line:
                inFunctionTableAdds = True
            elif inFunctionTableAdds:
                print(f'  {implFileName} \\')
                inFunctionTableAdds = False
                inFunctionTableAddsWritten = True
            print(line, end='')
        if inFunctionTableAdds or not inFunctionTableAddsWritten:
            raise Exception(f'Failure in adding {implFileName} to .lwmk file')

def main():
    parser = argparse.ArgumentParser(description='Generate a new OptiX 7 extension. '
        'The path of the intended extension implementation file needs to be given, as an absolute path. Both '
                                     'extension name and OptiX root directory will be extracted from it. E.g.\n\t'
                                     'python extensionGenerator.py C:\\Raven_bfm\\apps\\optix\\exp\\context\\ExtName.cpp',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('ExtensionImplFilePath')
    #parser.add_argument('-r', '--root', default=os.getcwd(), help='OptiX root directory, containing both "include" and "src" subdirs')
    parser.add_argument('-l', '--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARNING')
    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=args.level)
    filePath = args.ExtensionImplFilePath
    # don't use underscores...
    (pathName, fileName) = os.path.split(filePath)
    if not fileName:
        log.error(f'given argument \'{filePath}\' doesn\'t seem to contain a file name')
        return
    if fileName.find('_') != -1:
        fileName = fileName.replace('_', '')
        filePath = os.path.join(pathName, fileName)
        log.warning(f'Replaced underscores in file name and using now changed value {filePath}')

    if not os.path.isabs(filePath):
        filePath = os.path.join(os.getcwd(), filePath)
        log.info(f'started with relative directory. Expecting new extension file to be {filePath}')

    try:
        generator = ExtensionGenerator(filePath)
        generator.generateFiles()
    except Exception as err:
        log.error(err)
        return


if __name__ == '__main__':
    main()
