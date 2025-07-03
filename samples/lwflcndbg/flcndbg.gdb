define flcndbgll
    if !g_LwFlcnDbgModuleHandle
        # that is because somehow loading flcndbg 
        # makes g_ChiplibBusy equal to 0
        set $tmp = g_ChiplibBusy
        call 'Xp::LoadDLL'(g_LwFlcnDbgModuleName,&g_LwFlcnDbgModuleHandle,false)
        sharedlibrary libflcndbg.so
        p unmapFn = exitLwWatch
        # return the real value ot g_ChiplibBusy
        p g_ChiplibBusy = $tmp
    else
        echo The library was already loaded. Unload it first with flcndbgul.\n
    end
end

document flcndbgll
    Used to load the lwwatch shared library. To load it again you have to 
    unload it with flcndbgul.
end

define flcndbgul
    if g_LwFlcnDbgModuleHandle
        call exitLwWatch()
        p unmapFn = 0
        call 'Xp::UnloadDLL'(g_LwFlcnDbgModuleHandle)
        p g_LwFlcnDbgModuleHandle = 0
    else
        echo The library is not loaded.\n
    end
end

document flcndbgul
    Used to unload the lwwatch shared library.
end

define flcndbgsafe
    if g_ChiplibBusy > 0
        echo Looking for a safe point...\n\n
        b setChiplibBusy if value==-1 && g_ChiplibBusy==1
        #b RmIsr
        continue
        next
        clear setChiplibBusy
        #clear RmIsr
    else
        echo You are at a safe point.\n\n
    end
end

document flcndbgsafe
    Goes to the first possible place where lwwatch commands can be issued.
    This is connected with the problems that may occur with the fmodel 
    threads.
end

define flcndbg
    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call lwsafe to reach a safe point.\n
    else
        set $check = 0

        if $arg0 == init
            set $check = 1
            call init("")
        end

        if $check == 0
            echo lwwatch: no lw helper, passing arg1 to function...\n
            # No helper call direct assuming a string...
            call $arg0($arg1)
        end
    end
end

document flcndbg
    Used to issue flcndbg commands. Works only for some commands.
    Usage: flcndbg <command> <arguments>
end

define flcndbgv
    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call lwsafe to reach a safe point.\n
    else
        if $arg1 == 0
            call $arg0("")
        end

        if $arg1 == 1
            call $arg0("$arg2")
        end

        if $arg1 == 2
            call $arg0("$arg2 $arg3")
        end

        if $arg1 == 3
            call $arg0("$arg2 $arg3 $arg4")
        end

        if $arg1 == 4
            call $arg0("$arg2 $arg3 $arg4 $arg5")
        end

        if $arg1 == 5
            call $arg0("$arg2 $arg3 $arg4 $arg5 $arg6")
        end
    end
end

document flcndbgv
    Used to issue flcndbg commands. Needs number of arguments and list of arguments.
    Usage: flcndbgv <command> <number of arguments> <arguments>
end

define flcndbgs
    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call flcndbgsafe to reach a safe point.\n
    else
        call $arg0($arg1)
    end
end

document flcndbgs
    Used to issue lwwatch commands. Needs all the arguments to be put in quotes.
    Usage: flcndbgs <command> "<arguments>"
end
