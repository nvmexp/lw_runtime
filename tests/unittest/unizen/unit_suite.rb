#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#

#***************************************************************************#
#                        Module : unit_suite                                #
#       Class definition for various structure that store information       #
#       about the suites and methods to operate upon this information       #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'fileParseUtils'
require 'commonInfo'

# class to represent a suite in the unit test infra
class UnitSuite

	# Flag to check if suite is enabled or not
	attr_reader :isEnabled

	# Stores the name of the suite
	attr_reader :suiteName 

	# Stores the name of the suite in function to be called form
	attr_reader :suiteFunction 

	#
	# test source Path : the path at which unit test corresponding 
	# this Suite is Written
	#
	attr_reader :testSrcPath
	attr_writer :testSrcPath

	# Stores the list of files which contain the floating DOCs
	attr_reader :fileList

	#
	# Stores the path to header of the SUT : for static elimination 
	# this path is relative to chips_a
	#
	attr_reader :headerPath
	
	# return type of the SUT
	attr_reader :returnType

	# is the SUT corresponding to this suite static ?
	attr_reader :isStatic

	#
	# an array to store the names of test cases which are to
	# run and/or disabled
	#
	attr_reader :tcToRun, :tcToDisable
	attr_writer :tcToRun, :tcToDisable

	#
	# initializes the suite with name and default state disabled
	# also creates the string which would serve as the function call for this suite
	#
	def initialize(name, isEnabled = FALSE, returnType = nil, isStatic = nil, headerPath = nil, functionName = nil)

		@suiteName = name
		@suiteFunction = "suite_" + name if ! functionName
		testSrcPath = nil
		@isEnabled = isEnabled
		@fileList = []
		@returnType = returnType
		@isStatic = isStatic
		@headerPath = headerPath
		@tcToRun = []
		@tcToDisable = []

	end


	# Enable/ Disable the Suite
	def setEnabled(flag)

		@isEnabled = flag

	end

	# add a file/path to the file list
	def addToFileList(filePath)

		file = FileToBeWrapped.new(filePath)
		@fileList << (file)

	end


end # Unit Suite

# class to store information about "floating" DOCs to be mocked

class FileToBeWrapped

	# path of the file whic contains "floating" functions, relative to chips_a
	attr_reader :path

	# list of functions to be wrapped around
	attr_reader :doc

	# Name of this object when instantiated
	attr_reader :objectName

	# list of SUTs which are static
	attr_reader :sut

	def initialize(path)

		@path = path
		@doc = {}
		@sut = {}

		fileObjectName = @path.gsub("/", "_")
		fileObjectName = fileObjectName.sub(/\.c/, "")
		@objectName = fileObjectName.gsub(/\-/, "_")

	end

	def addDoc(ret, name, args = nil, isStatic = false)

		myDoc = DocDef.new(ret, name, args, isStatic)
		@doc[name] = (myDoc)

	end

	def addSut(ret, name)

		mySut = StaticSut.new(ret, name)
		@sut[name] = (mySut)

	end

	#
	# generate wrappers and headers for each Doc in this file
	# it also removes static keyword from static SUTs
	#
	def generateMocks(fileTouched = false)

		extractArgumentString

		filePath = @path
		filePath = filePath.gsub(/^\//,"")
		fileName = filePath.scan(/[\w\-]+\.c$/) 

		raise "Invalid File Path for wrapper generation" if fileName.length == 0
		
		oFile = "#{$unitBranch}/drivers/resman/arch/lwalloc/unittest/mocks/g_" + fileName[0]
		iFile = "#{$unitBranch}/" + filePath

		if File.exist?(oFile)
			readFile = oFile
			fileAlreadyExists = true
		else
			readFile = iFile
			fileAlreadyExists = false
		end

		if fileTouched
			readFile = iFile
			fileAlreadyExists = false
			system("rm #{oFile.sub(".c", ".h")}") if File.exist?(oFile.sub(".c", ".h"))
		end
		
		readBuffer = IO.read(readFile)
		
		# Eliminate static SUTs here
		@sut.each do |key, thisSut|

			starString = thisSut.returnType.scan(/\*+/).to_s
			myReturnType = thisSut.returnType.sub(starString, "")

			readBuffer.gsub!(/(static)\s+(#{myReturnType}\s+#{starString}#{thisSut.name}\s*\(([^)]+)\)\s*;)/, "\\2")
			readBuffer.gsub!(/(static)\s+(#{myReturnType}\s+#{starString}#{thisSut.name}\s*\(([^)]+)\)\s*\{)/, "\\2")

		end

		mockBuffer = String.new
		headerBuffer = String.new 

		if !fileAlreadyExists
			readBuffer = readBuffer.sub(/(\#include .+\n)/, "\\1
// Headers for mocking certain functions in this file, for Unit Testing
#include \"rmutmock.h\"
#include \"g_#{fileName[0].sub(".c", ".h")}\"

")
			headerMacro = fileName[0].sub(".c", "unit_h")
			headerMacro = headerMacro.gsub(/\-/, "_").upcase

			headerBuffer = "
/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef #{headerMacro}
#define #{headerMacro}

"
		end # file doesn't exist
		
		@doc.each do |key, doc|
			next if !readBuffer.gsub!(/(#{doc.name})(\s*\([^)]+\)\s*\{)/, "\\1_original\\2")

			# 
			# var storing the function prototype
			# for Ex: mockBuffer =
			#  RM_STATUS
			# foo
			# (
			# LwU32 someVar1
			# LwU32 someVar2
			# )
			#
			mockProto = String.new
			
			#
			# var storing the function call string 
			# for Ex: mockCallString = 
			# (someVar1, someVar2)
			#
			mockCallString = String.new
			#puts doc.argString
			type, name, ptrIndex = extractArgNameType(doc.argString)
			#puts type
			#puts name
			mockProto.appendOnNextLine!(doc.returnType) 
			mockProto.appendOnNextLine!(doc.name)
			mockProto.appendOnNextLine!("(")
			mockCallString += "("
			i = 0
			
			name.each do |argName|

				mockProto.appendOnNextLine!(type[i].to_s + ' ' + argName.to_s + ',')
				mockCallString += argName + ", "

				i += 1
				
			end

			mockCallString = mockCallString.gsub(/,\s$/, ")")

			mockProto += "%"
			mockProto = mockProto.gsub(/,%/, "\n)\n")

			mockBuffer.appendOnNextLine!("// #{doc.name} is wrapper around #{doc.name}_original for unit testing")

			mockBuffer += mockProto # function prototype
			headerBuffer += "static\n" if doc.isStatic
			headerBuffer += mockProto + ";"
			headerBuffer.appendOnNextLine!()
			headerBuffer.appendOnNextLine!()
			mockProto = mockProto.gsub(doc.name, " " + doc.name + "_unit")
			mockBuffer.appendOnNextLine!("// This is a comment which serves a purpose, please dont remove it")
			mockBuffer.appendOnNextLine!("{")
			mockBuffer.appendOnNextLine!("    typedef " + (mockProto.gsub("\n","")).gsub(/\s+/, " ") + ";")
			mockBuffer.appendOnNextLine!("    #{doc.name}_unit *unit_fp;")
			mockBuffer.appendOnNextLine!("    if ( !UTAPI_IsMocked(\"#{doc.name}\") )")

			# if this has a non-void return type
			if doc.returnType =~ /^void$/

				mockBuffer.appendOnNextLine!("        #{doc.name}_original#{mockCallString};")

			else

				mockBuffer.appendOnNextLine!("        return #{doc.name}_original#{mockCallString};")

			end

			mockBuffer.appendOnNextLine!("    else")
			mockBuffer.appendOnNextLine!("    {")
			mockBuffer.appendOnNextLine!("        unit_fp = (#{doc.name}_unit *) UTAPI_GetMockedFunction(\"#{doc.name}\", 1);")
			mockBuffer.appendOnNextLine!("        if (unit_fp != NULL)")
			mockBuffer.appendOnNextLine!("            return unit_fp#{mockCallString};")
			mockBuffer.appendOnNextLine!("        else")
			mockBuffer.appendOnNextLine!("        {")

			ptrIndex.each do |index|


				if type[index] =~ /(^LwU)|(^LwS)|(^LwB)|(^BOOL)|(^U0)/ and !(type[index] =~ /const/)

					mockBuffer.appendOnNextLine!("           if (UTAPI_IsParamMocked(\"#{doc.name}\", \"#{index+1}\"))")
					mockBuffer.appendOnNextLine!("               *#{name[index]} = (#{(type[index]).sub("*", "")}) UTAPI_MockReturnParam(\"#{doc.name}\", \"#{index+1}\");")

				end

			end

			unless doc.returnType =~ /^void$/
				mockBuffer.appendOnNextLine!("            return (#{doc.returnType}) UTAPI_MockReturn(\"#{doc.name}\");")    
			end

			mockBuffer.appendOnNextLine!("        }")
			mockBuffer.appendOnNextLine!("    }")
			mockBuffer.appendOnNextLine!("}")
			mockBuffer.appendOnNextLine!()
			mockBuffer.appendOnNextLine!()    

			type.clear
			name.clear
			ptrIndex.clear
			mockProto = ""
			mockCallString = ""

		end

		system("rm #{oFile}") if File.exist?(oFile)
		writeFile = File.new(oFile, "w+")

		fileAlreadyExists = true if File.exist?(oFile.sub(".c", ".h"))

		if !fileAlreadyExists
		
			headerFile = File.new(oFile.sub(".c", ".h"), "w+")
			headerFile.puts headerBuffer + "#endif\n"

		else
		
			headerReadBuffer = IO.read(oFile.sub(".c", ".h"))
			system("rm #{oFile.sub(".c", ".h")}")
			headerFile = File.new(oFile.sub(".c", ".h"), "w+")
			headerFile.puts headerReadBuffer.sub("#endif", headerBuffer + "#endif")
			
		end

		headerFile.flush

		writeFile.puts readBuffer
		writeFile.puts mockBuffer
		writeFile.flush

	end

	#
	# Extracts the argument strings for each DOC present in this file
	# Also, determines if the function is static 
	#
	def extractArgumentString

		iFile = "#{$unitBranch}/" + @path.gsub(/^\//,"")

		readBuffer = IO.read(iFile)

		@doc.each do |key, doc|

			starString = doc.returnType.scan(/\*+/).to_s
			returnType = doc.returnType.sub(starString, "")
	
			match = readBuffer.scan(/(static)?\s+#{returnType}\s+#{starString}#{doc.name}\s*\(([^)]+)\)\s*;/)
			if match[0]
				doc.isStatic = true if match[0][0] =~ /static/
			end

			match = readBuffer.scan(/(static)?\s+#{returnType}\s+#{starString}#{doc.name}\s*\(([^)]+)\)\s*\{/) # unless match[0]

			raise "Unable to parse function #{doc.name}() in file #{path}" unless match[0]

			doc.isStatic = true if match[0][0] =~ /static/

			match[0][1] = match[0][1].gsub(/\/\/.+\n/, "")
			match[0][1] = match[0][1].gsub(/\/\*.+\*\//, "")
			match[0][1] = match[0][1].gsub(/\n/, "")
			doc.argString = match[0][1]

		end

	end # extractArgumentString


	end # FileToBeWrapped

class DocDef

	# return type of function
	attr_reader :returnType

	# name of function
	attr_reader :name

	# argument string i.e. argument type and name
	attr_reader :argString
	attr_writer :argString

	# flag to check whether it's static 
	attr_reader :isStatic
	attr_writer :isStatic

	def initialize(ret, name, args = nil, flagStatic = false)

		@returnType = ret
		@name = name
		@argString = args
		@isStatic = flagStatic

	end

end # DocDef

class StaticSut

	# return type of function
	attr_reader :returnType

	# name of function
	attr_reader :name

	def initialize(ret, name)

		@returnType = ret
		@name = name

	end

end # StaticSut
