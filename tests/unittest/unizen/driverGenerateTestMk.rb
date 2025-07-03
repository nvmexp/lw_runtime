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
#                      Module : driverGenerateTestMk                        #
#    Serves as a front end to generateTestMk so that definition files       #
#                can be passed individually to generateTestMk               #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'commonInfo'

ARGV.each do |definition|

    system("#{$rubyApp} #{$unizenPath}/generateTestMk.rb #{definition}")

end
