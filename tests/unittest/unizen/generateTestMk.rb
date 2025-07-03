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
#                        Module : generateTestMk                            #
#  Generates makefile which specifies which all tests are to be compiled    #
#        A Makefile will be generated for every definition file             #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'commonInfo.rb'
require '../suitedefs/parents.rb'

#
# this is the definition file, for which
# a corresponding makefile would be generated
#
# Everytime this is triggered a new makefile corresponding 
# this definition file would be generated
#
defFile = ARGV[0]

# ensure the file is in proper format
raise "Invalid input Definition File, Must be in the <filename>.rb format" unless defFile =~ /\w+\.rb/

# here we load the input definition file
require "#{defFile}"

#$resman.disableCompile

# generates the contents of the makefile
def generateMakefileContents(thisSuite)

	writeBuffer = "# Unit Test Source for Suites under #{thisSuite.node.suiteName}\n\n"

	# this(2nd cond) checks if this is disabled by means of an ancestor being disabled
	if thisSuite.bCompile == true and thisSuite.shouldBeCompiled? == true

		#
		# loop around each sub suite of the main suite of the
		# definition file
		#
		thisSuite.each do |suite|

			next if suite == thisSuite

			if suite.bCompile

				writeBuffer += "unitTestSrc += $(BRANCH)/" + suite.node.testSrcPath + "\n"

			end

		end

	end

	return writeBuffer

end

makeFileDir = "#{$unitBranch}/drivers/resman/arch/lwalloc/unittest/#{$buildType}/makefiles"
makeFilePath = makeFileDir + "/#{$thisSuite.node.suiteName}.mk"

File.delete(makeFilePath) if File.exist?(makeFilePath)
wFile = File.open(makeFilePath, "w+")

# write the contents to the file
wFile.puts generateMakefileContents($thisSuite)

wFile.flush
wFile.close



