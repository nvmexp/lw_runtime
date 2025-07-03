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
#                        Module : generateParents                           #
#         A program that produces a defintion file from the directory       #
#                           structure of the folder                         #
#                                                                           #
#***************************************************************************#
require 'find'

kernelDir = 'C:/p4/sw/dev/gpu_drv/chips_a/drivers/resman/kernel'

genDir = 'C:/p4/sw/dev/gpu_drv/chips_a/diag/unittest/common/testrunner/suitedefs'
Dir.chdir(kernelDir)
kernelModules = Dir.glob("**")

kernelModules.delete("inc")

fileHeader = String.new
fileSubHeader = String.new
fileBody  = String.new

star = String.new()
blank = String.new()
79.times { star += '#'} 
25.times { blank += ' '}



fileCopyright = \
"#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#"


writeFile = File.new("parents.rb", "w")

writeFile.puts fileCopyright

kernelModules.each do |mod| 

	fileHeader += "\n\n\ndir_#{mod} = (SuiteTree.new(\"dir_#{mod}\", TRUE)) \n\n"  

	fileBody += "\n\n\n$kernel.add(dir_#{mod})\n\n"

	Dir.chdir(kernelDir + '/' + mod)

	subModules = Dir.glob("**")

	# eliminate files, if present 
	subModules.each { |dir| subModules.delete(dir) if FileTest.file?(dir) }

	subModules.each do |subMod|

		fileHeader += "    dir_#{mod}_#{subMod} = (SuiteTree.new(\"dir_#{mod}_#{subMod}\", TRUE)) \n"

		fileBody += "    dir_#{mod}.add(dir_#{mod}_#{subMod})\n"

	end

end



writeFile.puts "\n#" + star
writeFile.puts blank + " Suite Creation Section " + blank 
writeFile.puts "#" + star
writeFile.puts fileHeader
writeFile.puts "\n#" + star
writeFile.puts blank + " Suite Addition Section " + blank 
writeFile.puts "#" + star
writeFile.puts fileBody








