!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX R1, f[TEX5], TEX5, 2D;
TEX R0, f[TEX5], TEX5, 2D;
DP3R_SAT R1.xyz, R1, R0;
TEX R0, f[TEX2], TEX2, 2D;
MULR R1.xyz, R1, R1;
MULR R1.w, R0, C0;
ADDR_SAT R1.w, R1, C1;
MULR R1.xyz, R1, R1;
ADDR_m4_SAT R1.w, R1, R1;
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.xyz, R1.w, R0;
MULR R1.xyz, R1, R1;
MULR R0.xyz, R1, R0;
MULR R0.xyz, f[COL0], R0;
END

# Passes = 8 

# Registers = 2 

# Textures = 3 
