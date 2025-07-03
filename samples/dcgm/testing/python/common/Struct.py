# Allows for easily creating "anonymous" objects
# From http://norvig.com/python-iaq.html
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)
