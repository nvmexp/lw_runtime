!!FP1.0
DECLARE C0={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MULR R0.xyz, R1, R0;
ADDR R0.w, {1, 1, 1, 1}, -R1;
MULR R0.xyz, C0, R0;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
