define lwll
    if !g_LwWatchModuleHandle
        # that is because somehow loading lwwatch
        # makes g_ChiplibBusy equal to 0
        set $tmp = g_ChiplibBusy
        call 'LwWatchLoadDLL'()
        sharedlibrary liblwwatch.so
        p unmapFn = exitLwWatch
        # return the real value ot g_ChiplibBusy
        p g_ChiplibBusy = $tmp
    else
        echo The library was already loaded. Unload it first with lwul.\n
    end
end

document lwll
    Used to load the lwwatch shared library. To load it again you have to
    unload it with lwul.
end

define lwul
    if g_LwWatchModuleHandle
        call exitLwWatch()
        p unmapFn = 0
        call 'LwWatchUnloadDLL'()
        p g_LwWatchModuleHandle = 0
    else
        echo The library is not loaded.\n
    end
end

document lwul
    Used to unload the lwwatch shared library.
end

define lwsafe
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

document lwsafe
    Goes to the first possible place where lwwatch commands can be issued.
    This is connected with the problems that may occur with the fmodel
    threads.
end

define lw
    if $argc == 0
        return
    end

    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call lwsafe to reach a safe point.\n
    else
        set $check = 0

    if $argc == 1
        call $arg0("")
    end
    if $argc == 2
        call $arg0("$arg1")
    end
    if $argc == 3
        echo 3 args
        call $arg0("$arg1 $arg2")
    end
    if $argc == 4
        call $arg0("$arg1 $arg2 $arg3")
    end
    if $argc == 5
        call $arg0("$arg1 $arg2 $arg3 $arg4")
    end
    if $argc == 6
        call $arg0("$arg1 $arg2 $arg3 $arg4 $arg5")
    end
    if $argc == 7
        call $arg0("$arg1 $arg2 $arg3 $arg4 $arg5 $arg6")
    end
    if $argc == 8
        call $arg0("$arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7")
    end
    if $argc == 9
        call $arg0("$arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8")
    end
    if $argc == 10
        call $arg0("$arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $arg9")
    end

end

document lw
    Used to issue lwwatch commands.
    Usage: lw <command> <arguments>
end

define lwv
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

document lwv
    Used to issue lwwatch commands. Needs number of arguments and list of arguments.
    Usage: lwv <command> <number of arguments> <arguments>
end

define lws
    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call lwsafe to reach a safe point.\n
    else
        call $arg0($arg1)
    end
end

document lws
    Used to issue lwwatch commands. Needs all the arguments to be put in quotes.
    Usage: lws <command> "<arguments>"
end

define rv
    if g_ChiplibBusy == 1
        echo Cannot execute command now. Call lwsafe to reach a safe point.\n
    else    
        if $argc == 0
            call rv("")
        end

        if $argc == 1
            call rv("$arg0")
        end

        if $argc == 2
            call rv("$arg0 $arg1")
        end

        if $argc == 3
            call rv("$arg0 $arg1 $arg2")
        end

        if $argc == 4
            call rv("$arg0 $arg1 $arg2 $arg3")
        end

        if $argc == 5
            call rv("$arg0 $arg1 $arg2 $arg3 $arg4")
        end
    end
end

document rv
    RISC-V handling functions.
end
