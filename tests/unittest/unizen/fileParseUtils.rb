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
#                       Module : fileParseUtils                             #
#             Some utilities for parsing/generating files                   #
#                                                                           #
#***************************************************************************#

def extractArgNameType(argString)

	name = []
	type = []
	ptrIndex = []
	index = 0

	nameBuffer = []
	typeBuffer = []
	argList = argString.split(',')
	
	if argList.length == 1 
	
		if argList[0] =~ /^\s*void\s*$/
		
		return type, name, ptrIndex
		
		end
		
	end

	argList.each do |arg|

		nameBuffer = arg.scan(/((\**)\w*(?:\w|(?:\s+))$)/)
		typeBuffer = arg.gsub(nameBuffer[0][0].to_s, "")

		ptrIndex.push(index) if typeBuffer.scan("*").length > 0

		if nameBuffer[0][1] != ""

			typeBuffer+=  (nameBuffer[0][1]).to_s
			nameBuffer[0][0] = (nameBuffer[0][0]).gsub(nameBuffer[0][1], "")
			ptrIndex.push(index)

		end



		name.push((nameBuffer[0][0]).gsub("\n","").strip)
		type.push(typeBuffer.gsub("\n","").strip)

		index += 1

	end 

	raise "Number of arguments: type/name mismatch" if name.length != type.length
	return type, name, ptrIndex

end # extractArgNameType


# Extend the String class to Append the text on new line
class String 

	def appendOnNextLine!(text = "")

		self << "\n" + text

	end

end # String


