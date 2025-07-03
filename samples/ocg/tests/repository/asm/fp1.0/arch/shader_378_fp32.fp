!!FP1.0
DECLARE C1={ 0.4, 0.5, 0.34, 0.45 };
DECLARE C4={ 0.7, 0.65, 0.45, 0.57 };
TEX R0, f[TEX0], TEX0, 2D;
MULR R1.xyz, f[COL0], C1;
MULR R0.xyz, R0, R1;
MULR R0.xyz, R0, C4;
MULR R0.w, R0.w, C1.w;
ADDR H0.xyz, R0, R0;
MOVR H0.w, R0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
