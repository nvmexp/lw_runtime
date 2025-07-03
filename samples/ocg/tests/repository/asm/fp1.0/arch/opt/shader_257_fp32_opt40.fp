!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX3], TEX3, 2D;
DP3R_SAT R0.xyz, R0, R1;
TEX R1, f[TEX2], TEX2, 2D;
MOVR R0.w, R1;
MULR R0.xyz, R0, R0;
ADDR R0.w, R0, C0;
MULR_SAT R0.w, R0, C1;
MULR R0.xyz, R0, R0;
ADDR_m4_SAT R0.w, R0, R0;
TEX R1, f[TEX0], TEX0, 2D;
MULR R1.xyz, R0.w, R1;
MULR R1.xyz, f[COL0], R1;
MULR R1.xy, R0, R1;
MULR R1.w, R0.xywz, R1.xywz;
MULR R0.xyz, R1.xywz, R0;
END

# Passes = 8 

# Registers = 2 

# Textures = 3 
