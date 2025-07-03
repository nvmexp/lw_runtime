!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MULR R2.xyz, R0, f[COL0];
MOVR R0.w, f[COL0].w;
MULR R2.xyz, R1, R2;
MULR R2.xyz, C0, R2;
MULR R2.xyz, R2, {2, 0, 0, 0}.x; 
MULR R3, C1, R0;
MADR R4.xyz, R0.w, -R2, R2;
MADR R0.xyz, R0.w, R3, R4;
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 5 

# Textures = 2 
