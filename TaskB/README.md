# TaskB

# Key Assumption - I am assuming that the code is being run on Standard_NC24 Azure VM. This assumption has been made because the organizers have mentioned that this is machine which will be used to execution. Since this machine contains 4 GPUs, I have hardcoded different GPUs in my run scripts and hence all of these runs can be executed in parallel. In case the VM is changed, organizers will need to make appropriate changes in the code.

# Time taken to complete one run  - Approx 24 hours

To run TaskB, follow the given steps - 

1. chmod 700 install.sh
2. chmod 700 activate.sh
3. chmod 700 decode_TaskB_run1.sh
4. chmod 700 decode_TaskB_run2.sh
5. chmod 700 decode_TaskB_run3.sh
6. ./install.sh
7. ./activate.sh
8. ./decode_HealthMavericks_run1.sh
9. ./decode_HealthMavericks_run2.sh
10 ./decode_HealthMavericks_run3.sh
