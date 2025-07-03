!!FP2.0 
TEX R1, f[TEX5], TEX5, 2D;
TEX R0, f[TEX5], TEX5, 2D;
DP3R_SAT R1.xyz, R1, R0;
TEX R0, f[TEX2], TEX2, 2D;
MOVR_SAT R1.w, R0;
MULR R1.xyz, R1, R1;
TEX R0, f[TEX0], TEX0, 2D;
ADDR_m4_SAT R1.w, R1, R1;
MULR R1.xyz, R1, R1;
MULR R0.xyz, R1.w, R0;
MULR R1.xyz, R1, R1;
MULR R0.xyz, f[COL0], R0;
MULR R1.xyz, R1, R1;
MULR R0.xyz, R1, R0;
END

# Passes = 8 

# Registers = 2 

# Textures = 3 
