-- an implementation of printf

function printf(...)
 io.write(string.format(...))
end

printf("Hello %s from %s on %s\n",os.getelw"USER" or "there",_VERSION,os.date())
