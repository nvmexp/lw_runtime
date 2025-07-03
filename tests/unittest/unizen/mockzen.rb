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
#                        Module : mockzen                                   #
#       A driver program that accepts the file and generates mocks for      #
#           all the DOCs in the file that it can find from database         #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'commonInfo'

p4root =String.new( ELW["P4ROOT"])
p4root = p4root.gsub("\\","/")

# dont exactly remember why this was needed 
# after people start using this and nothing bad happens
# this dead code would be removed

#alreadyInDesiredDir = true
#lwrrentWorkingDir = Dir.pwd
#unless lwrrentWorkingDir =~ /unittest\/unizen/
#	Dir.chdir("#{p4root}/sw/dev/gpu_drv/chips_a/diag/unittest/unizen")
#	alreadyInDesiredDir = false
#end

raise "\n\nCan't proceed further Generated Database(generatedDocDb.rb) not available" unless File.exist?("#{$unizenPath}/generatedDocDb.rb")

# dont exactly remember why this was needed 
# after people start using this and nothing bad happens
# this dead code would be removed

#unless alreadyInDesiredDir
#	Dir.chdir(lwrrentWorkingDir)
#end

require 'generatedDocDb'

thisPath = String.new(ARGV[0]) # defreeze the input argument

raise "\n\nIlwalid input Source File, Must be in the <filename>.c format" unless thisPath =~ /\w+\.c/

thisPath = (thisPath.scan(/drivers\/.+/))[0]

raise "\n\nError : Database for this file #{thisPath} doesn't exist." unless $fileList[thisPath]

if ARGV.length == 2 and ARGV[1] =~ /-fileTouched/

	$fileList[thisPath].generateMocks(true)

else

	$fileList[thisPath].generateMocks()

end

