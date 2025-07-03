for i in {0..35};
    do
     echo "&&&& RUNNING dcgmi_diag_test"
     ./../dcgmi diag -r 3 -p "diagnostic.test_duration=600;sm stress.test_duration=600;targeted power.test_duration=600;targeted stress.test_duration=600"
     if [ $? -eq 0 ]
     then
         echo "&&&& PASSED dcgmi_diag_test"
     else
         echo "&&&& FAILED dcgmi_diag_test"
     fi
    done
