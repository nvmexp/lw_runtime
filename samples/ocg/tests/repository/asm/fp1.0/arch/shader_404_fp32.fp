!!FP1.0
TEX R0, f[TEX2], TEX1, 2D;
TEX R1, f[TEX0], TEX0, 2D;
MULR R1.xyz, R1, f[COL0];
MULR R1.w, R1, f[COL0];
MULR R1.xyz, R0, R1;
MULR R1.xyz, R1, {1.987654, 0, 0, 0}.x;
MOVR R0, R1;
MOVR o[COLR], R0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
