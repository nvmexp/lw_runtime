!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
DP3R R0.xyz, R0, R1;
MOVR R0.w, R1;
MULR R0, R0, C0;
ADDR_SAT R0, R0, C1;
MULR R0.xyz, R0, f[COL0];
ADDR_m4_SAT R0.w, R0, R0;
MADR R0.xyz, R0, R0.w, {0, 0, 0, 0};
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
