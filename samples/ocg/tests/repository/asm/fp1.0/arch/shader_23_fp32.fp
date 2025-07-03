!!FP1.0 
DECLARE C0 = {0.9, 0.8, 0.7, 0.5};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MULR R0, R0, f[COL0];
MULR R0.xyz, R1, R0;
MULR H0.xyz, C0, R0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
