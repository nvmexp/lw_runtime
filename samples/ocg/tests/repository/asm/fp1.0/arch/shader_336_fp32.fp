!!FP1.0
DECLARE C0={0.5, 0.5, 0.5, 0.5};
TEX R3, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
ADDR_SAT R0.w, {0, 0, 0, 1}, -R2.w;
MADR R0, R0.w, -R3, R3;
MADR R0, R0.w, R1, R0;
MULR R0, R0, R2;
MULR H0.xyz, C0, R0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 8 

# Registers = 4 

# Textures = 3 
