

The Unit Tests in this folder are for the APIs exported to the user of this infrasrtucture.

Tested APIs are

1.	UNIT_INSTALL_READ_CALLBACK : This allows a read operation on a register to be modelled in any specific         manner as the user wants. When ever a gpu read operation is performed then user installed function is called         to return the value as computed by this function
2.	UNIT_INSTALL_READ_RETURN_ALWAYS : This allows for a same value to be returned every a particular register is         read.
3.	UNIT_INSTALL_READ_RETURN_UNTIL_COUNT : This allows for a particular value to be reurned for N conselwtive         number of times.
4.	UNIT_READ_VALUE_ON_NTH_WRITE :This allows for reading the value which was written on Nth write operation         for a particular register.
5.	UNIT_READ_REGISTER :This allows for reading the last value written on the register.
6.	UNIT_INSTALL_WRITE_CALLBACK: This allows for modelling the write beahvior of the register. User specified         function is called for evry gpu write.
7.	UNIT_INSTALL_WRITE_MIRROR: This allows for a value written on register R1 to be mirrored(written) on         register R2.

And redirected Gpu read/write operations

1.      unitGpuWriteRegister032
2.      unitGpuReadRegister032
