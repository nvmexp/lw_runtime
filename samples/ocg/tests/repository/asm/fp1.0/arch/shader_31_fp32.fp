!!FP1.0 
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX1], TEX1, 2D;
TEX R1, f[TEX2], TEX2, 2D;
MULR R0, R0, R1;
MULR R0.xyz, C2, R0;
MULR H0, R0, f[COL0];
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
