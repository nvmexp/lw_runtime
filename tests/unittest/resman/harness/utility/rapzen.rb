
#
# creates a wrapper around every function in the given file
# generates 2 files one containg the original + wrappers and 
# conatining the associated header
# iFile is the input file to be parsed
# oFile is the path/name of output file that would be generated
# if oFile is not indicated then the file is created with the same name as
# the input file in the unittest/debug directory
#
def generateWrapper(iFile, oFile=nil)

 a = Time.now
 
 # read the entire file and save it in this buffer
 fileBuffer = IO.read(iFile)
 raise "Invalid File Name or Path" if fileBuffer.length <=0

 # string containing the wrapper code for the given function
 wrappers = String.new
 # string containing the function signature
 names = [] 
 
 # array of all function names
 allFunctionNames = []
 
 #
 # integer value indicating the position of pointer var in the 
 # function prototype
 #
 ptr = []

 #
 # discard all single line comments, really a big pain in the neck
 # output file would be devoid of all comments
 # better to look in original file for the comments
 #
 fileBuffer = fileBuffer.gsub(/\/\/.*\n/,"\n")
 fileBuffer = fileBuffer.gsub(/\/\*.*?\*\//,"\n")
 
 # discard all code between #if 0 & #endif
 
 # Dead code for debugging
 #tempFile = File.new("temp.c", "w")
 #
 tempBuffer = fileBuffer.split("\n")
 fileBuffer = ""
 inifdef = false
 tempBuffer.each do |line|
  
  if line.scan(/#if\s0/).length > 0 
   inifdef = true
  end
  if !inifdef
   fileBuffer += line +"\n"
  else
   if line.scan(/#endif/).length > 0 
     inifdef = false
   end
   # dead code for debugging
   #tempFile.puts line + "\n"
   #
  end
 end

 # dead code for debugging
 #tempFile.puts fileBuffer
 #
 
 #
 # extract out all the function definition signature
 # 0: function signature
 # 1: return type
 # 2: function Name
 # 3: argument list along with type
 #
 fileBuffer.scan(/(((?:static\s+)?(?:const\s+)?\w+(?:\s*static)?(?:\s*__cdecl)?(?:\s*\*+)?)\s+(\*?\w+)\s*\(([^)]*)\)\s*\{)/).each do |match|

 # save off the function name
 functionName = match[2].to_s
 returnType = match[1].to_s
 argumentList = match[3].to_s
 
  
 # make sure it's not a C keyword ie if/else/while/for etc
 if ( functionName.scan(/if/).length == 0 && functionName.scan(/else/).length == 0 && functionName.scan(/for/).length == 0 && functionName.scan(/while/).length == 0 && functionName.scan(/switch/).length == 0 && returnType.scan(/define/).length == 0 && argumentList.scan(/F064/).length == 0 && returnType.scan(/^if/).length == 0 && returnType.scan(/__cdecl/).length == 0)

  # save off the function signature here, "names" contains all of them
  names.push(match[0]) if !names.include?(match[0])
  
  # ensure that this function wasn't parsed before 
  next if allFunctionNames.include?(functionName)
 
  # save the function name in the list
  allFunctionNames.push(functionName)
  
  # arrays to store the name and corresponding type for the arguments
  argType = []
  argName = []
  #
  # integer value indicating the position of pointer var in the 
  # function prototype
  #
  ptrs = []

  # split the argument string into individual arguments
  args = match[3].split(',')

  # A variable to store the index of the ptr argument
  i=0

  #
  # colwert int *foo() to int* foo(), so that we get to know 
  # that return type is pointer
  #  
  namePtr = match[2].scan(/^\*+/).to_s
  if namePtr.length != 0
   match[2] = match[2].gsub(namePtr,"")
   match[1] += namePtr
  end

  args.each {|arg|

   # extract the argument name 
   name = arg.scan(/\s(?:\*+)?\w*\w(?:\[\w*\])?(?:\s+)?$/)
   # extract the type
   type = arg.gsub(name.to_s,"")

   # dead code for debugging
   #tempFile.puts "before -name:#{name}   type:#{type}\n"
   #

   #
   # colwert "type *name" to "type * name"
   # if earlier type = LwU32 & name = *someName
   # the, now type = LwU32* & name = someName
   #
   ptrName = name.to_s.scan(/\*+/).to_s
   name = name.to_s.gsub(ptrName, "")
   type += ptrName

   # if this is ptr variable then save off the index
   ptr.push(i) if ptrName.length != 0 or type.scan("*").length >0

   # if name has "[]" then strip it off
   name = name.gsub(/\[\w*\]/, "")
   # save the name and type, avoid "..." for va_list
   if type.scan("...").length == 0
   argName.push(name)
   argType.push(type)
   end

   i += 1;

   # dead code for debugging
   #print "after -name:#{name}   type:#{type}\n"
   #

   }

   #
   # a variable to store the callstring i.e (a,b,c)
   # as against (int a, int b, int c)
   #
   callString = "("
   argName.each do |name|
    name = "" if name.scan("void").length != 0
    callString += "#{name}," 
   end
  
   if argName.length > 0
    callString = callString.gsub(/,$/, ");") 
   else
    callString += ");"
   end    
   

   # a variable to check if the function returns something
   doesReturn = false
   doesReturn = true if match[1].scan(/void/).length == 0

   # eliminate static from return type
   retWOStatic = match[1].gsub("static", "")


   # this is the code that goes into each wrapper
   wrappers += match[1] + " " + match[2]+"(" +match[3] +")\n{\n"
   wrappers+= "    typedef #{retWOStatic} #{match[2].upcase + "_FP"} (#{match[3]});\n"
   wrappers +="    #{match[2].upcase + "_FP"} *unitFunc;\n"
   wrappers +="    if ( !UTAPI_IsMocked() )\n"
   if doesReturn
    wrappers +="        return #{match [2]}_original#{callString}\n"           
   else
    wrappers +="        #{match [2]}_original#{callString}\n"           
   end
   wrappers +="    else\n"
   wrappers +="    {\n"
   wrappers +="        unitFunc = (#{match[2].upcase + "_FP"} *) UTAPI_GetMockedFunction(1);\n"
   wrappers +="        if (unitFunc != NULL)\n"
   wrappers +="        {\n"
   wrappers +="            return unitFunc#{callString}\n" 
   wrappers +="        }\n"
   wrappers +="        else\n"
   wrappers +="        {\n"   

   ptr.each do |index|
    wrappers +="            if (UTAPI_IsParamMocked(#{index+1}))\n"
    wrappers +="            {\n"
    wrappers +="                #{argName[index]} = (#{argType[index]}) (UTAPI_MockReturnParam(#{index+1}));\n"
    wrappers +="            }\n"
   end

   wrappers +="            return (#{retWOStatic}) UTAPI_MockReturn();\n" 
   wrappers +="         }\n"
   wrappers +="    }\n"
   wrappers += "}\n\n"
   
   argName.clear
   argType.clear
   ptr.clear

  end
 end

 # a variable to store the name of the header file to be generated.
 fileHeader = String.new
 names.each{ |name| 

  # this goes into the header file
  fileHeader += name.gsub(/\n\{/,";\n")   

  # this changes the function definitions to "_original" versions 
  replaceName = name. gsub(/(\w+)(\s*)\(/,"\\1_original\\2(")
  fileBuffer = fileBuffer.gsub(name,replaceName)
 
 }

 if oFile == nil
  fileName = iFile.scan(/\w+\.c$/) 
  
  # dead  code for debugging
  #print fileName
  #
  
  p4root =String.new( ELW["P4ROOT"])
  p4root.gsub("\\","/")
  print "\n p4root"
  print p4root
  p4root = p4root + "/sw/dev/gpu_drv/chips_a/drivers/resman/arch/lwalloc/unittest/debug/"
  oFile = p4root + fileName[0]
  #oFile = oFile.gsub(".c","unit.c")
 end
 system("rm #{oFile}") if File.exist?(oFile)
 writeFile = File.new(oFile, "w+")
 hFile = oFile.gsub(".c","unit.h")
 system("rm #{hFile}") if File.exist?(hFile)
 writeFileHeader = File.new(hFile, "w+")
 raise "Unable to create new file:#{oFile}" if writeFile == nil
 print hFile

 # this includes the generated header file into the generated source file
 if names.length > 0
  firstFooDefinition = names[0]. gsub(/(\w+)(\s*)\(/,"\\1_original\\2(") 
  fileBuffer = fileBuffer.gsub(firstFooDefinition, "\n\n#include \"#{hFile}\"\n\n#{firstFooDefinition}")
end

 # here we write the buffers on to the files
 writeFileHeader.puts fileHeader
 writeFile.puts fileBuffer
 writeFile.puts "\n\n#include \"rmutmock.h\"\n\n"
 writeFile.puts wrappers

 b = Time.now
 
 print "\n\nWrapper File generated: #{oFile} in #{b-a} seconds\n"
end

input1 = ARGV[0]
input2 = ARGV[1]

# replace backslash with forwardslash
input1 = input1.gsub(/\\/,"/")

if input2 == nil
  
 generateWrapper(input1)
 
else
  
 input2 = input2.gsub(/\\/,"/")    
 generateWrapper(input1, input2)
 
end

