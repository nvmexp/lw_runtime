import os.path
import platform

def find_magic_string (file, pattern):

    _data  = open (file, "rb", -1).read()


    _start = _data.find (str (pattern), 0)

    if _start < 0:
        return None

    _end = _data.find (chr (0), _start)
    if _end < 0:
        _end = len (_data)

    return unicode (_data[_start:_end])
    
def detect_exelwtable(dir, filename, extension):
	default = os.path.join(dir, filename + extension)
	if platform.system() == "Darwin" and not os.path.isfile(default):
		return unicode(os.path.join(dir, filename + ".app", "Contents", "MacOS", filename))
	else:
		return unicode(default)
