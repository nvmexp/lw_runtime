!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R2, f[TEX0], TEX0, 2D;
TEX R3, f[TEX1], TEX1, 2D;
TEX R4, f[TEX2], TEX2, 2D;
TEX R5, f[TEX3], TEX3, 2D;
MULR R0, R2, f[COL0];
ADDR R5.w, {1, 1, 1, 1}, -R5;
MULR R1, R4, R5.w;
MADR R0.xyz, R1, C2, R0;
MULR R0.xyz, R3, R0;
MULR H0.xyz, C0, R0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 8 

# Registers = 6 

# Textures = 4 
