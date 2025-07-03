
# Add all the individual test scripts to this master script file to execute them all

#script arguments
    #$1 = Product Model used for verification

arg1="$1"

python3 test_basic_sanity.py $arg1
python3 fm_process_start_stop_test.py
python3 p2p_write_ce_bandwidth_test.py
python3 fm_non_fatal_error_injection.py
python3 fm_fatal_error_injection.py

