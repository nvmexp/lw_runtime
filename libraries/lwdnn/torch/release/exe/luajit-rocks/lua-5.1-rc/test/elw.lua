-- read environment variables as if they were global variables

local f=function (t,i) return os.getelw(i) end
setmetatable(getfelw(),{__index=f})

-- an example
print(a,USER,PATH)
