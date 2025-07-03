import gdb

print('Hello from lwwatch.py!')
print('Commands can be ilwoked with \'lwpy <command> [args]\'.')
print('Arguments prefixed with #, * or & are evaluated based on the current GDB frame.')
print('Set your breakpoint and run: lwpystart [Bar0Addr] to load lwwatch and continue to the breakpoint.')


#
# Helpers
#

# Exception raised when parsing errors occur
class ParseException(Exception):
    def __init__(self,message="Error parsing arguments."):
        self.message = message

# Check if chiplib is busy
def isChiplibBusy():
    g_ChiplibBusy = gdb.parse_and_eval('g_ChiplibBusy')
    if g_ChiplibBusy != 0:
        print('Cannot execute command now. Call lwsafe to reach a safe point.')
        return True
    else:
        return False

# Parse a string of arguments
def parseArguments(argString):
    argSplit = argString.split(' ')
    argList = []

    argiSplit = filter(lambda s : s != '', argSplit) # remove empty strings resulting from splitting on double whitespace

    for i in range(len(argSplit)):
        if argSplit[i] == ' ':
            argSplit.pop(i)

    length = len(argSplit)
    if length == 0:
        return ''
    i = 0
    while(i < length):
        if len(argSplit[i]) > 2 and argSplit[i][0] == '#' and argSplit[i][1] == '{':
            foundMatch = False
            statement = []
            for j in range(i, length):
                position = argSplit[j].find('}')
                if position == len(argSplit[j]) - 1:
                    statement.append(argSplit[j])
                    foundMatch = True
                    i = i + j
                    break
                elif position == -1:
                    statement.append(argSplit[j])
                else:
                    print('Unexpected evaluation terminator found at ' + argSplit[j])
            if foundMatch == False:
                print('Unmatched {, check your input')
                
            try:
                statement = ' '.join(statement)
                statement = statement.replace('{', '')
                statement = statement.replace('}', '')
                argList.append(parseArgument(statement))
            except gdb.error:
                print("err")
                raise ParseException('Unable to evaluate ' + ' '.join(statement).replace('{', '').replace('}', ''))
        else:
            try:
                argList.append(parseArgument(argSplit[i]))
            except gdb.error:
                raise ParseException(argSplit[i])
        i += 1

    return ' '.join(argList)


# Parse symbol and return the value as string
def parseArgument(arg):
    parsedArg = None
    if len(arg) == 0:
        return arg

    if arg[0] == '*' or arg[0] == '#' or arg[0] == '&':
        # Drop the hash char if its present and evaluate the variable
        if arg[0] == '#':
            arg = arg[1:]
        parsedArg = gdb.parse_and_eval(arg)
        return hex(parsedArg)
    return arg

#
# Gdb command definitions
#

# Used to load the lwwatch shared library. To load it again you have to unload it with lwul.
class lwll(gdb.Command):
    def __init__(self):
        super(lwll, self).__init__('lwll', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        if gdb.parse_and_eval('g_LwWatchModuleHandle') == 0:
            bIsChiplibBusy = gdb.parse_and_eval('g_ChiplibBusy')
            gdb.execute('call LwWatchLoadDLL()')
            exitLwWatch = gdb.parse_and_eval('exitLwWatch').address.cast(gdb.lookup_type('unsigned long long'))
            print(gdb.parse_and_eval('unmapFn = ' + str(exitLwWatch)))
            gdb.parse_and_eval('g_ChiplibBusy = ' + str(bIsChiplibBusy))
        else:
            print('The library was already loaded. Unload it first with lwul.')

# Used to unload the lwwatch shared library.
class lwul(gdb.Command):
    def __init__(self):
        super(lwul, self).__init__('lwul', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        if gdb.parse_and_eval('g_LwWatchModuleHandle') != 0:
            gdb.execute('call exitLwWatch()')
            gdb.parse_and_eval('unmapFn = 0')
            gdb.execute('call LwWatchUnloadDLL()')
            gdb.parse_and_eval('g_LwWatchModuleHandle = 0')
        else:
            print('The library is not loaded.')

class lwpystart(gdb.Command):
    def __init__(self):
        super(lwpystart, self).__init__('lwpystart', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        arg = parseArguments(arg)

        gdb.execute('break Gpu::Initialize')
        gdb.execute('run')
        gdb.execute('lwll')
        gdb.execute('lwpy init ' + arg)
        gdb.execute('clear Gpu::Initialize')
        gdb.execute('continue')

# Goes to the first possible place where lwwatch commands can be issued.
# This is connected with the problems that may occur with the fmodel
# threads.
class lwsafe(gdb.Command):
    def __init__(self):
        super(lwsafe, self).__init__('lwsafe', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        if isChiplibBusy() == True:
            print('Looking for a safe point...')
            gdb.execute('break setChiplibBusy if value == -1 && g_ChiplibBusy == 1')
            gdb.execute('continue')
            gdb.execute('next')
            gdb.execute('clear setChiplibBusy')
        else:
            print('You are at a safe point.')

# Used to issue lwwatch commands.
# Usage: lw <command> <arguments>
class lwpy(gdb.Command):
    def __init__(self):
        super(lwpy, self).__init__('lwpy', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        if isChiplibBusy() == True:
            return
        if len(arg) == 0:
            print('No command provided.\nUsage: lwpy <command> [args]')
            return
        cmd = arg.split(' ')[0]
        try:
            args = arg.split(' ')
            args.pop(0)
            args = ' '.join(args)
            args = parseArguments(args)
        except ParseException as err:
            print('ParseException: ' + err.message)
            return

        print(cmd + ' ' + args)
        gdb.execute('call ' + cmd +'(\"' + args + '\")')

# RISC-V handling functions.
class rv(gdb.Command):
    def __init__(self):
        super(rv, self).__init__('rv', gdb.COMMAND_DATA)

    def ilwoke(self, arg, from_tty):
        if isChiplibBusy() == True:
            return
        gdb.execute('call rv(\'' + arg + '\')')

# Boilerplate code to register the command classes to the gdb runtime at 'source' time.
lwll()
lwul()
lwsafe()
lwpy()
lwpystart()
rv()
