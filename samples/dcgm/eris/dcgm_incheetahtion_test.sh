for i in {0..38};
    do
     echo "&&&& RUNNING dcgm_integration_test"
      python main.py --no-lint --dvssc-testing
     if [ $? -eq 0 ]
     then
         echo "&&&& PASSED dcgm_integration_test"
     else
         echo "&&&& FAILED dcgm_integration_test"
     fi
    done
