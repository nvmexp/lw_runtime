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
#                        Module : genext                                    #
#       generates ruby extensions for all suites that it knows about        #
#                     irrespective of their run status                      #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'unit_suite'
require 'suite_tree'
require 'fileParseUtils'
require "#{$unitPath}/suitedefs/parents"

# get the build environment
$buildType = ELW["BUILD_CFG"]
$buildType = 'debug' unless $buildType
out_dir = "#{$unitBranch}/drivers/resman/arch/lwalloc/unittest/#{$buildType}"

$:.unshift(out_dir)

dbBuffer = String.new

# load the database only, if it exists
if File.exist?("#{out_dir}/g_extDb.rb")
	require 'g_extDb'
else
	$extList = {}
end

ARGV.each do |defFile|

	# ensure the file is in proper format
	raise "Invalid input Definition File, Must be in the <filename>.rb format" unless defFile =~ /\w+\.rb/

	# here we load the input definition file
	require "#{defFile}"

	defName =defFile.scan(/\/([^\/]+)\.rb/)[0].to_s
	
	# assuming if one doesnt exist, all three don't exist
	unless File.exist?("#{out_dir}/exp_#{defName}.c")

		allTestsBuffer = String.new($cCopyright)
		cExportsBuffer = String.new($cCopyright)
		hExportsBuffer = String.new($cCopyright)

		allTestsBuffer.appendOnNextLine!
		cExportsBuffer.appendOnNextLine!
		hExportsBuffer.appendOnNextLine!

		allTestsBuffer.appendOnNextLine!("//all suite creation functions go here")
		hExportsBuffer.appendOnNextLine!("#define ADD_TEST_#{defName} ")

	else

		allTestsBuffer = String.new
		cExportsBuffer = String.new
		hExportsBuffer = String.new

	end

	isDirty = false

	$thisSuite.each do |suite|

		if suite.children.length == 0 and !$extList[suite.node.suiteName]
			allTestsBuffer.appendOnNextLine!("LwSuite* #{suite.node.suiteFunction}();")

			hExportsBuffer += "\\\n    rb_define_module_function(mUnizen, \"#{suite.node.suiteFunction}\", _wrap_#{suite.node.suiteFunction}, -1);"

			cExportsBuffer.appendOnNextLine!("SWIGINTERN VALUE")
			cExportsBuffer.appendOnNextLine!("_wrap_#{suite.node.suiteFunction}(int argc, VALUE *argv, VALUE self) {")
			cExportsBuffer.appendOnNextLine!("  LwSuite *result = 0 ;")
			cExportsBuffer.appendOnNextLine!("  VALUE vresult = Qnil;")
			cExportsBuffer.appendOnNextLine!
			cExportsBuffer.appendOnNextLine!("  if ((argc < 0) || (argc > 0)) {")
			cExportsBuffer.appendOnNextLine!("    rb_raise(rb_eArgError, \"wrong # of arguments(%d for 0)\",argc); SWIG_fail;")
			cExportsBuffer.appendOnNextLine!("  }")
			cExportsBuffer.appendOnNextLine!("  result = (LwSuite *)#{suite.node.suiteFunction}();")
			cExportsBuffer.appendOnNextLine!("  vresult = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_LwSuite, 0 |  0 );")
			cExportsBuffer.appendOnNextLine!("  return vresult;")
			cExportsBuffer.appendOnNextLine!("fail:")
			cExportsBuffer.appendOnNextLine!("  return Qnil;")
			cExportsBuffer.appendOnNextLine!("}")
			cExportsBuffer.appendOnNextLine!
			cExportsBuffer.appendOnNextLine!

			dbBuffer.appendOnNextLine!("$extList[\"#{suite.node.suiteName}\"] = true")

			isDirty = true

		end

	end

	if isDirty

		allTests = File.new("#{out_dir}/allTest_#{defName}.h", "a")
		hExports = File.new("#{out_dir}/exp_#{defName}.h", "a")
		cExports = File.new("#{out_dir}/exp_#{defName}.c", "a")
		rbExtDb = File.new("#{out_dir}/g_extDb.rb", "a")

		# update the generated files
		allTests.puts allTestsBuffer
		allTests.flush
		allTests.close

		hExports.puts hExportsBuffer
		hExports.flush
		hExports.close

		cExports.puts cExportsBuffer
		cExports.flush
		cExports.close

		rbExtDb.puts dbBuffer
		rbExtDb.flush
		rbExtDb.close

	end

end # ARGV.each

