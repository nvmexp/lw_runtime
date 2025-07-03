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
#                        Module : generateSuiteDef                          #
#              generates code for suite definition files                    #
#                                                                           #
#***************************************************************************#

require 'fileutils'
$:.unshift File.dirname(__FILE__)

require 'unit_suite.rb'
require 'suite_tree.rb'
require 'fileParseUtils'

# following function is applicable only for linux environment
def colorMe(color, str)

	unless (RUBY_PLATFORM =~ /win/)

		return color + str + "\e[0m"

	else

		return str

	end

end

def colorMeGreen(str)

	colorMe("\e[32m", str)

end

def colorMeBlue(str)

	colorMe("\e[36m", str)

end


def getCorrectResponse()

	response = gets.strip.downcase

	if response.eql?("yes") or response.eql?("no")

		return response

	else

		puts colorMeGreen("only Yes/No is acceptable\n")
		getCorrectResponse

	end

end

# should have created a single function here

def moreFilesResponse()

	puts colorMeGreen("Are there Any more Files with dependent(floating) Functions(Yes/No)")
	return getCorrectResponse

end

def moreDocsResponse()

	puts colorMeGreen("Are there Any more dependent(floating) Functions in this File(Yes/No)")
	return getCorrectResponse

end

def hasHeaderResponse()

	puts colorMeGreen("Does it have an assosciated Header where declaration is static(Yes/no):")
	return getCorrectResponse

end

def haveDocResponse()

	puts colorMeGreen("Does it have dependent(floating) functions ?")
	return getCorrectResponse

end

#
# returns the SUT name and
# source file where SUT exists
#
def getBasicInfo

	puts colorMeGreen("Enter the name of the Suite to be added, should be the name of SUT")
	suiteName = gets.strip

	puts colorMeGreen("Enter Return Type of the SUT")
	returnType = gets.strip

	puts colorMeGreen("Enter the path to File  where the SUT exists(relative to Branch/chips_a) 
	for example \"drivers/resman/kernel/fb/fermi/fbgf100.c\"")

	sutPath =  gets.strip
	sutPath.sub!(/\\/,"/")
	sutPath.sub!(/^\//,"")

	raise "Error : SUT should exist in a .c source" unless sutPath =~ /\.c/

	suiteSrcPath = nil

	if sutPath =~ /\/resman\/(.+)/

		resmanRelativePath = $1
		suiteSrcPath = "diag/unittest/resman/suites/" + resmanRelativePath.sub(/\.c/, "")

	end

	iFile = "#{$unitBranch}/" + sutPath
	readBuffer = IO.read(iFile)

	starString = returnType.scan(/\*+/).to_s
	myReturnType = returnType.sub(starString, "")

	matchRegex = /(static)?\s+#{myReturnType}\s+#{starString}#{suiteName}\s*\(([^)]+)\)\s*\{/
	raise "\n#{suiteName}() not Found in #{sutPath}.\nPlease try again!" unless readBuffer =~ matchRegex

	isStatic = false

	if $1

		isStatic = true 

	# can't tell if it's static at this point, check for header declaration
	else

		match = readBuffer.scan(/(static)?\s+#{myReturnType}\s+#{starString}#{suiteName}\s*\(([^)]+)\)\s*;/)

		if match[0]

			isStatic = true if match[0][0]

		end

		# else, it's definitely not static

	end

	return suiteName, sutPath, returnType, isStatic, suiteSrcPath

end


#
# generates a template source file where
# unit tests can be added
#
def generateTestTemplate(suiteName)

	FileUtils::mkdir_p("#{$unitPath}/temp")

	testSrcPath = "#{$unitPath}/temp/ut_#{suiteName}.c"

	return if File.exist?(testSrcPath)

	writeFile = File.new(testSrcPath, "w+")

	writeBuffer = "
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
*/
"

	writeBuffer.appendOnNextLine!("#include \"rmunit.h\"")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("
/*
 * @brief Test Suite where to add all test cases for fifoServiceTop_GF100
*/")
	writeBuffer.appendOnNextLine!("LwSuite* suite_#{suiteName}()")
	writeBuffer.appendOnNextLine!("{")
	writeBuffer.appendOnNextLine!("    LwSuite* suite = UTAPI_NEW_SUITE(NULL, NULL);")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("    UTAPI_ADD_TC(suite,")
	writeBuffer.appendOnNextLine!("                 (TestFunction)<Test Case 1 Setup Function>,")
	writeBuffer.appendOnNextLine!("                 (TestFunction)<Test Case 1 exercise function>,")
	writeBuffer.appendOnNextLine!("                 (TestFunction)<Test Case 1 Teardown Function>);")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("// More Test Case can be added here in similar Fashion")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("    return suite;")
	writeBuffer.appendOnNextLine!("}")
	
	writeBuffer.appendOnNextLine!("
/*
 * @brief Setup function for test case :<Test case name/Details> 
 *
 * @param[in] tc  Test Case pointer
 *
*/
")
	writeBuffer.appendOnNextLine!("void <Test Case 1 Setup Function>(LwTest *tc)")
	writeBuffer.appendOnNextLine!("{")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("    // Your Setup code")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("}")

		writeBuffer.appendOnNextLine!("
/*
 * @brief Teardown function for Test Case :<Test case name/Details> 
 *
 * @param[in] tc  Test Case pointer
 *
 */
")
	writeBuffer.appendOnNextLine!("void <Test Case 1 Teardown Function>(LwTest *tc)")
	writeBuffer.appendOnNextLine!("{")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("    // Your Teardown code")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("}")

			writeBuffer.appendOnNextLine!("
/*
 * @brief Test Case 1 <Name> : <Details>
 *
 * @param[in] tc  Test Case pointer
 *
*/
")
	writeBuffer.appendOnNextLine!("void <Test Case 1 exercise function>(LwTest *tc)")
	writeBuffer.appendOnNextLine!("{")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("// Test Case Body")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("//")
	writeBuffer.appendOnNextLine!("// Refer ")
	writeBuffer.appendOnNextLine!("// https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Architecture/RM_Test_Infrastructure/RM_Unit_Test_Framework/Dolwmentation_on_RM_unit_Test_Infra")
	writeBuffer.appendOnNextLine!("// for information on various APIs/Features")
	writeBuffer.appendOnNextLine!("//")
	writeBuffer.appendOnNextLine!("")
	writeBuffer.appendOnNextLine!("}")

	writeFile.puts writeBuffer
	writeFile.flush
	writeFile.close
	puts colorMeBlue("Test Source file Template generated")
	puts colorMeBlue(testSrcPath)

end


#
# interrogates the user for information
# needed to create a new suite
#
def generateDefinitionContent

	suiteName, sutPath ,retType, isStatic, suiteSrcPath = getBasicInfo

	thisSutPath = sutPath

	fileName = sutPath.scan(/\/([\w\-]+)\.c/)[0][0]
	sutInFile = "file_" + fileName
	definitionFilePath = sutPath.sub("/#{fileName}.c", "")
	sutPath = definitionFilePath
	parentSuite = String.new

	definitionFilePath = "#{$unitPath}/suitedefs/#{definitionFilePath}/#{fileName}.rb"

	parentSuite = sutPath.scan(/[\w\-]+\/[\w\-]+\w$/)[0]
	parentSuite.sub!("/", "_")
	parentSuite = "dir_" + parentSuite 

	# here check if a definition for this file exists also check if there is an appropriate entry in parents.rb
	File.exist?(definitionFilePath) ? definitionPresent = 1 : definitionPresent = 0
	(IO.read("#{$unitPath}/suitedefs/parents.rb").scan(parentSuite).length >= 1) ? parentPresent = 1 : parentPresent = 0

	# check, if this suite already exists
	suiteDefBuffer = IO.read(definitionFilePath) if definitionPresent == 1

	suiteAlreadyExists = false
	suiteAlreadyExists = true if suiteDefBuffer =~ /SuiteTree.new\("#{suiteName}"/

	if suiteAlreadyExists

		puts colorMeGreen("This suite Already exists, using the new Information which you would enter now,
		The suite Defintion would be updated\n")

	end

	#
	# for future use, to eliminate static
	# lwrrently, not used
	# 
	sutHeaderPath = "nil"

	#hasHeader = hasHeaderResponse

	# lwrrently disabled
	hasHeader = "no"

	if hasHeader =~ /yes/

		puts colorMeGreen("Enter the path of header where it's declared static(relative to Branch/chips_a)")
		sutHeaderPath = gets.strip

		if sutHeaderPath =~ /none/
			sutHeaderPath = "nil"
		end

	end

	mySuite = (SuiteTree.new(suiteName.to_s, true, retType, isStatic, sutHeaderPath))

	#
	# bit 0 is definitionPresent, bit 1 is parentPresent
	# case 0 : Cretae a new def and add parent in parents.rb
	# case 1 : This is an error 
	# case 2 : Create a new def
	# case 3 : Append to the defintion file
	#
	taskToBeDone = 2*parentPresent + definitionPresent

	raise "Error : The definition File for this Module Exists 
	\nwhile, there is no corresponding entry in the parent definition file" if taskToBeDone == 1

	index = 0

	moreFiles = haveDocResponse()

	while moreFiles.eql?("yes")
		puts colorMeGreen("Enter File Path realtive to Branch/chips_a which contains dependent floating functions:")
		myDocPath = gets.strip
		myDocPath.sub!(/\\/,"/")
		myDocPath.sub!(/^\//,"")
		mySuite.node.addToFileList(myDocPath)

		moreDocs = "yes"

		while moreDocs.eql?("yes")
			puts colorMeGreen("Enter Dependent Function's Name:")
			name = gets.strip

			puts colorMeGreen("Enter Dependent Function's Return Type:")
			returnType = gets.strip

			mySuite.node.fileList[index].addDoc(returnType, name)

			moreDocs = moreDocsResponse()

		end

		moreFiles = moreFilesResponse()
		index += 1
	end

	# reuse the variable 
	index = 0

	#
	# Following code creates the ruby code required  
	# to create and add a new suite into the 
	# Unit testing Infra
	#

	writeBuffer = "
	# @@#{suiteName}@@ BEGINS
	# create a new suite with name #{suiteName}
	#{suiteName.sub("-", "_")} = (SuiteTree.new(\"#{suiteName}\", TRUE, \"#{retType}\",  #{isStatic}, #{sutHeaderPath.to_s}))
	"

	sutFilePathCounter = nil
	# to determine if this file is already a dependent file(having DOCs)
	thisFileAlreadyExists = nil

	mySuite.node.fileList.each do |file|

	if file.path.casecmp(thisSutPath) == 0

		sutFilePathCounter = index
		thisFileAlreadyExists = true

	end

	writeBuffer += " 
	# add a new dependent(conatining bare function) file for #{suiteName}
	#{suiteName}.node.addToFileList(\"#{file.path}\")
	"

	writeBuffer += "
	# add the DOCs for this file "

	file.doc.each do |key, doc|


	writeBuffer += "
	#{suiteName}.node.fileList[#{index}].addDoc(\"#{doc.returnType}\", \"#{doc.name}\")
	"

	end


	index += 1

	end

	if isStatic == true

	unless thisFileAlreadyExists
	
	sutFilePathCounter = index 
	writeBuffer += " 
	# add a new dependent(conatining bare function or a static SUT) file for #{suiteName}
	#{suiteName}.node.addToFileList(\"#{thisSutPath}\")
	"
		writeBuffer += "
	#{suiteName}.node.fileList[#{sutFilePathCounter}].addSut(\"#{retType}\", \"#{suiteName}\")
	"

	else

		writeBuffer += "
	#{suiteName}.node.fileList[#{sutFilePathCounter}].addSut(\"#{retType}\", \"#{suiteName}\")
	"

	end # thisFileAlreadyExists

	end # isStatic


	writeBuffer += "
	# add the new suite #{suiteName} to its parent suite
	#{sutInFile.sub("-", "_")}.add(#{suiteName})
	# @@#{suiteName}@@ ENDS
	"

	if taskToBeDone == 0 or taskToBeDone == 2

		unless suiteSrcPath

			puts colorMeGreen("Enter the path to FOLDER where the Unit Tests would exist(relative to diag/unittest) 
			for example \" #{thisSutPath.sub(/\.c/, "").sub(/(^\w+\/)/, "\\1suites/")} \"")

			suiteSrcPath =  gets.strip
			suiteSrcPath.sub!(/\\/,"/")
			suiteSrcPath.sub!(/^\//,"")
			suiteSrcPath = "diag/unittest/" + suiteSrcPath

		end

		temp = "myParent = $resman.find(\"#{parentSuite}\") 

	raise \"\\n Error : Parent Suite can't be Found \\n\" if myParent == nil

	# the path of the folder where unittests of this suite would be present
	suiteSrcPath = \"#{suiteSrcPath}\"
	"

	defName = sutInFile.gsub("file_", "").sub("-", "_")

		temp += "
	#{sutInFile.sub("-", "_")} = (SuiteTree.new(\"#{sutInFile}\", TRUE))
	"

		writeBuffer  = temp + writeBuffer

		writeBuffer += "




	# @@Common_Def_Section@@
	# All suite additions must happen before the above comment i.e prior to Common_Def_Section

	# add the file level suite to Dir level suite
	myParent.add(file_#{defName})

	$thisSuite = file_#{defName}
	
	#
	# update the unit test source location for all tests
	# if any test doesn't reside in this expected Location
	# then override after this loop
	#
	$thisSuite.each do |test|

		test.node.testSrcPath = suiteSrcPath + \"/ut_\" + test.node.suiteName + \".c\"

	end
	
	#
	# refer https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Architecture/RM_Test_Infrastructure/RM_Unit_Test_Framework/Dolwmentation_on_RM_unit_Test_Infra
	# for more information on defintion generation/modification
	#
	"

	end


	if taskToBeDone == 3

		if suiteAlreadyExists
		
			match = /#\s+@@#{suiteName}@@\s+BEGINS.+#\s+@@#{suiteName}@@\s+ENDS/m
			suiteDefBuffer.sub!(match, writeBuffer)

		else

			match = /(#\s+@@Common_Def_Section@@)/
			suiteDefBuffer.sub!(match, "#{writeBuffer}\n\t\\1")

		end

		`rm #{definitionFilePath}`
		defFile = File.new(definitionFilePath, "w+")
		defFile.puts suiteDefBuffer
		defFile.flush
		defFile.close

		checked_out = `p4 opened`

		system("p4 open #{definitionFilePath}") unless checked_out =~ /#{sutInFile.gsub("file_", "")}\.rb/

	else

		FileUtils.mkdir_p(definitionFilePath.sub(/\/[\w\-]+\.rb$/, ""))
		defFile = File.new(definitionFilePath, "w+")
		defFile.puts writeBuffer
		defFile.flush
		defFile.close

		system("p4 add #{definitionFilePath}")

		if taskToBeDone == 0

			puts colorMeGreen("\nYou need to update the parents.rb refer Unit Test Wiki on how to do that")
			puts colorMeGreen("https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Architecture/RM_Test_Infrastructure/RM_Unit_Test_Framework/Dolwmentation_on_RM_unit_Test_Infra")

		end

	end

	puts colorMeBlue("\nSuite Definition File generated")
	puts colorMeBlue(definitionFilePath)

	generateTestTemplate(suiteName)

end # generateDefinitionContent


# call the function to generate code
generateDefinitionContent()
