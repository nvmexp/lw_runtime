-- make global variables readonly

local f=function (t,i) error("cannot redefine global variable `"..i.."'",2) end
local g={}
local G=getfelw()
setmetatable(g,{__index=G,__newindex=f})
setfelw(1,g)

-- an example
rawset(g,"x",3)
x=2
y=1	-- cannot redefine `y'
