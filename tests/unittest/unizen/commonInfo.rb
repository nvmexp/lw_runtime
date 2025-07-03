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
#                        Module : commonInfo                                #
#       Some common symbols to be used across all ruby source files         #
#                                                                           #
#                                                                           #
#***************************************************************************#


# gloabal to store the path to diag/unittest directory
$unitPath = File.expand_path(__FILE__).sub(/\/unizen.+/, "")

# global to store the path to branch directory
$unitBranch = $unitPath.sub(/\/diag\/unittest.*/, "")

# global holding patht to unizen folder
$unizenPath = $unitPath + "/unizen"

# gloabl holding address to P4ROOT elw var.
$p4root =String.new( ELW["P4ROOT"])
$p4root = $p4root.gsub(/\\/,"/")

$cCopyright = String.new("
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
*/
")

$rCopyright = String.new("
#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#
")

# 'rubyApp' to hold the path to ruby exelwtable
if RUBY_PLATFORM =~ /win/

	$rubyApp = $p4root + '/sw/tools/ruby1.8.6/bin/ruby.exe'

		unless File.exist?($rubyApp)

			$rubyApp = ELW["BUILD_TOOLS_DIR"] + "/ruby1.8.6/bin/ruby.exe"

		end

else

	if 1.size*8 == 64

		$rubyApp = $p4root + '/sw/tools/linux/ruby1.8.6/bin/ruby'

		unless File.exist?($rubyApp)

			$rubyApp = ELW["BUILD_TOOLS_DIR"] + "/linux/ruby1.8.6/bin/ruby"

		end

	elsif 1.size*8 == 32

		$rubyApp = $p4root + '/sw/tools/linux/ruby1.8.6/redhat/bin/ruby'

		unless File.exist?($rubyApp)

			$rubyApp = ELW["BUILD_TOOLS_DIR"] + "/linux/ruby1.8.6/redhat/bin/ruby"

		end

	else

		raise "Unable to find appropriate ruby binary"

	end

	raise "Unable to find appropriate ruby binary" unless File.exist?($rubyApp)

end

# get the build environment
$buildType = ELW["BUILD_CFG"]
$buildType = 'debug' unless $buildType
