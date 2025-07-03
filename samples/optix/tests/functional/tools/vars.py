import platform

if platform.system() == "Windows":
    EXE_EXTENSION = ".exe"
elif platform.system() == "Darwin":
    EXE_EXTENSION = ""
elif platform.system() == "Linux":
    EXE_EXTENSION = ""
else:
    # Better to throw an exception than silently choose something when we don't know what
    # the problem.
    raise Exception("Unknown platform: %s" % (platform.system(),) )
