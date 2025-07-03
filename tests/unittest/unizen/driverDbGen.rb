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
#                      Module : driverDbzen                                 #
#         Serves as a front ned to dbzen so that definition files           #
#                can be passed individually to dbzen                        #
#                                                                           #
#***************************************************************************#

$:.unshift File.dirname(__FILE__)

require 'commonInfo'

ARGV.each do |definition|

    system("#{$rubyApp} #{$unizenPath}/dbzen.rb #{definition}")

end
