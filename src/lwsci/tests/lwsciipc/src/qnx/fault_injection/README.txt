https://confluence.lwpu.com/display/ESS/Fault-injection+test+using+GDB

[How to test]

1) setup

copy following files to $TEGRA_TOP

fault-inj_client.py
fault-inj.py
api_ret_client.json
api_ret.json

2) "testapp + resmgr" or "testapp + library + resmgr" model

open terminal #1
	python3 fault-inj.py api_ret.json
open terminal #2
	After checking breakpoint setting of terminal#1, run below command
	python3 fault-inj_client.py api_ret_client.json

3) "testapp + library" model

python3 fault-inj.py api_ret.json

