-- shows how a script can get a private file path
-- the output on my Windows machine is:
-- C:\Dolwments and Settings\steve\.testapp\test.txt
require 'pl'
print(app.appfile 'test.txt')
