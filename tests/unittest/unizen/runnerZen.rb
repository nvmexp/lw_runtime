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
#                        Module : runnnerZen                                #
#                    Runs the suites as per user's Input                    #
#                                                                           #
#***************************************************************************#

require 'getoptlong'

$:.unshift File.dirname(__FILE__)

require 'unit_suite'
require 'suite_tree'
require 'commonInfo'
require '../suitedefs/parents'

# get the build environment
$buildType = ELW["BUILD_CFG"]
$buildType = 'debug' unless $buildType
out_dir = "#{$unitBranch}/drivers/resman/arch/lwalloc/unittest/#{$buildType}"

$:.unshift(out_dir)
$:.unshift($unitBranch)
require 'g_allDefs'

# globals to store the contents that would go into generated runner 
$functionBody = String.new
$functionHeader = String.new

# get the build environment
$runspace = ELW["UNIT_RUNSPACE"]
$runspace = "#{$unitPath}/#{$buildType}" unless $runspace


# disable a set of test cases of a given Suite
def disableTestCases(thisSuite)

	if 0 < thisSuite.node.tcToDisable.length

		$functionHeader += thisSuite.node.suiteName+" = Unizen::#{thisSuite.node.suiteFunction}\n"

	end

	thisSuite.node.tcToDisable.each do |tCase|

		$functionBody    += thisSuite.node.suiteName + "_" + tCase + \
			" = Unizen::GetTestCase(#{thisSuite.node.suiteName}, \"#{tCase}\")\n"
		$functionBody    += "raise \"#{tCase} : No such Test Case\" if !" + thisSuite.node.suiteName + "_" + tCase + "\n"
		$functionBody    += thisSuite.node.suiteName + "_" + tCase +".bSkip = 1\n\n"

	end

end # disableTestCases

# enable a set of test cases and disable all others for a given suite
def enableTestCases(thisSuite)

	if 0 < thisSuite.node.tcToRun.length

		$functionHeader += thisSuite.node.suiteName+" = Unizen::#{thisSuite.node.suiteFunction}\n"
		$functionBody    += "# disable all the test cases for this Suite\n"
		$functionBody    += "Unizen::SkipAllTests(#{thisSuite.node.suiteName})\n"

	end

	thisSuite.node.tcToRun.each do |tCase|

		$functionBody    += "#{thisSuite.node.suiteName}_#{tCase} = Unizen::GetTestCase(#{thisSuite.node.suiteName}, \"#{tCase}\")\n"
		$functionBody    += "raise \"#{tCase} : No such Test Case\" if !" + thisSuite.node.suiteName + "_" + tCase + "\n"
		$functionBody    += "#{thisSuite.node.suiteName}_#{tCase}.bSkip = 0\n\n"

	end

end # enableTestCases


#
# generates the contents of runner script specific 
# to a part of the tree which is represented by "suite"
# and everything under it
#
# also enables the test cases, if any as specified by
# the user
#
def enableAndAddSuites(suite, fileHandle)

	if suite.node.isEnabled

		if suite.children.length > 0

			$functionHeader += suite.node.suiteName + " = Unizen::LwSuiteNew(nil, nil, \"\")\n"
			$functionBody  += "Unizen::LwSuiteAddSuite(motherSuite, "+suite.node.suiteName+")\n\n"

		else

			enableTestCases(suite)
			unless 0 < suite.node.tcToDisable.length or 0 < suite.node.tcToRun.length

				$functionBody    += "Unizen::LwSuiteAddSuite(motherSuite, Unizen::"+suite.node.suiteFunction+");\n"

			else

				$functionBody    += "Unizen::LwSuiteAddSuite(motherSuite, "+suite.node.suiteName+")\n"

			end

		end

	end

	suite.children.each do |subSuite|

		next unless subSuite.node.isEnabled

		subSuite.forEachEnabled do |subSubSuite|

			if subSubSuite.children.length > 0

				$functionHeader += subSubSuite.node.suiteName+" = Unizen::LwSuiteNew(nil, nil, \"\")\n"
				$functionBody    += "Unizen::LwSuiteAddSuite("+subSubSuite.parent.node.suiteName+", "+subSubSuite.node.suiteName+")\n"

			else

				unless subSubSuite.parent.node.suiteName =~ /dir_/

					enableTestCases(subSubSuite)

					unless 0 < subSubSuite.node.tcToDisable.length or 0 < subSubSuite.node.tcToRun.length

						$functionBody    += "Unizen::LwSuiteAddSuite("+subSubSuite.parent.node.suiteName+", Unizen::"+subSubSuite.node.suiteFunction+");\n"

					else

						$functionBody    += "Unizen::LwSuiteAddSuite("+subSubSuite.parent.node.suiteName+", "+subSubSuite.node.suiteName+")\n"

					end

				end

			end

		end

		if subSuite.node.isEnabled

			$functionBody      += "\n"  

		end

	end

end # enableAndAddSuites


#
# generates the complete script which will 
# be used to run the requested tests
#
def generateRunner( fileHandle )

		fileHandle.puts "$:.unshift File.dirname(__FILE__)"
		fileHandle.puts "require 'unizen'"
		fileHandle.puts ""
		fileHandle.puts "output = Unizen::LwStringNew()"
		fileHandle.puts "
###############################################################################
#                                Suite Declaration                            #
###############################################################################
	"
		fileHandle.puts

		fileHandle.puts "motherSuite = Unizen::LwSuiteNew(nil, nil, \"\")\n"
		
		fileHandle.puts $functionHeader
		fileHandle.puts "
###############################################################################
#                                Suite Addition                               #
###############################################################################
	"
		fileHandle.puts $functionBody
		fileHandle.puts "
###############################################################################
#                                Suite Run                                    #
###############################################################################
	"
		fileHandle.puts "Unizen::LwSuiteRun(motherSuite, 1)"
		fileHandle.puts "# test reprots will be available here"
		fileHandle.puts "Unizen::LwSuiteDetails(motherSuite, output)"
		fileHandle.puts "puts output.buffer"
		fileHandle.flush
		fileHandle.close

end # generateRunner


#
# verify the hierarchy as mentioned, is correct or not,
#  as represented by the defintion files we have
#
def verifyHierarchy(hrchy, root)

	if hrchy.class == Array

		length = (hrchy.length) -1

	else

		length = 0

	end


	if (length == 0) 

		if ( root.node.suiteName =~ /#{hrchy.to_s}/ )

			return true, root 

		else

			return false, nil

		end

	end


	root.children.each do|suite| 

		if suite.node.suiteName =~ /#{hrchy[1]}/

			hrchy.delete_at(0)
			return verifyHierarchy(hrchy, suite)
			break

		end

	end

	return false, nil

end # verifyHierarchy


#
# creates the tree from the command line
# Enable/Disables the suite Irrespective of what their 
# status is in definition files
#
# "flag" to determine whether to enable(true) or disable(false)
#
def createTreeCommandLine(suiteString, tcArr, writeFile, flag)

	suitesHrchy = suiteString.split(":")

	hrchyRoot = $unit.find(suitesHrchy[0])

	if hrchyRoot == nil

		if suitesHrchy[0] =~ /suite_/

			raise "\nError:Invalid Hierarchy: #{suiteString}. Please Try with out \"suite_\"\n"

		else

			raise "\nError:Invalid Hierarchy: #{suiteString}\n"

		end

	end

	verifyStatus, runRoot = verifyHierarchy(suitesHrchy.clone,hrchyRoot)
	raise "Incorrct Hierarchy - " + suiteString if ! verifyStatus

	if flag

		runRoot.node.setEnabled(true)

		if tcArr

			tcArr.each do |tCase|

				runRoot.node.tcToRun.push(tCase)

			end

		end

		enableAndAddSuites(runRoot, writeFile)

	else

		runRoot.node.setEnabled(false) if !tcArr or tcArr.length == 0

		if tcArr

			tcArr.each do |tCase|

				runRoot.node.tcToDisable.push(tCase)

			end

		end

	disableTestCases(runRoot)

	end

end # createTreeCommandLine


#
# parses the options to determine
# : separated string of suites and 
# set of test cases
#
def parseOptions(arg)

	split = arg.split("~")
	raise "Invalid argument: #{arg}" unless split.length <= 2
	suites = split[0]
	tcs = split[1]
	tcsAr = tcs.split(",") if tcs

	return suites, tcsAr

end # parseOptions

if File.exist?("#{$runspace}/testRunner.rb")
	`rm #{$runspace}/testRunner.rb`
end

runnerLocation = "#{$runspace}/testRunner.rb"
writeFile = File.new(runnerLocation, "w")

opts = GetoptLong.new(
      [ '--skip', GetoptLong::REQUIRED_ARGUMENT ],
      [ '--run', GetoptLong::REQUIRED_ARGUMENT ],
      [ '--help', '-h', GetoptLong::NO_ARGUMENT ],
      [ '--gdb', '-g', GetoptLong::NO_ARGUMENT ]
    )

#
# just to let it know that this was just help 
# and no runner should be generated
#
bHelp = nil

# to tell if gdb is to be ilwoked
gdb = nil

# Arrays to hold --skip and --run arguments
skipArgs = []
runArgs = []

#
# this is based on the assumption that all --skip 
# will be placed before --run 
#
opts.each do |opt, arg|

	case opt

		when '--help'
		
			puts 
			puts "--gdb     Ilwoke the gdb for debugging"
			puts
			puts "--help    Print this message"
			puts
			puts "--run     Specify which suite/test case to run for example --run fb:fermi:fbgf100:fbAlloc_GF100~tc1,tc2"
			puts "          would run test case \"tc1\" and \"tc2\" of suite fballocgf100 of the specified hierarchy"
			puts 
			puts "--skip    Specify which suite/test case to skip for example --skip fb:fermi:fbgf100:fbAlloc_GF100~tc1,tc2"
			puts "          would skip test case \"tc1\" and \"tc2\" of suite fballocgf100 of the specified hierarchy and "
			puts "          run all other suites specified in --run"
			puts

			puts "Please ensure that are no spaces in the arguments of any options"
			puts 
			bHelp = true

		when '--run'

			runArgs.push(arg)

		when '--skip'

			skipArgs.push(arg)

		when '--gdb'

			gdb = true

	end

end


# process the skip arguments
skipArgs.each {|arg| 

	suitesString, tcArr = parseOptions(arg)
	createTreeCommandLine(suitesString, tcArr, writeFile, false)

}

# process the run arguments
runArgs.each {|arg| 

	suitesString, tcArr = parseOptions(arg)
	createTreeCommandLine(suitesString, tcArr, writeFile, true)

}

# create the runner file and Run the tests
unless bHelp
 
	generateRunner(writeFile)

	# run the tests

	if gdb

		system("gdb --args #{$rubyApp} #{runnerLocation}")

	else

		system("#{$rubyApp} #{runnerLocation}")

	end

end
