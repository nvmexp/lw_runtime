def find_magic_string (file, pattern):

    _data  = open (file, "rb", -1).read()


    _start = _data.find (str (pattern), 0)

    if _start < 0:
        return None

    _end = _data.find (chr (0), _start)
    if _end < 0:
        _end = len (_data)

    return unicode (_data[_start:_end])
