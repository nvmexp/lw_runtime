!!FP1.0 
DEFINE C1={1.0, 1.0, 1.0, 1.0};
DEFINE C6={6.0, 6.0, 6.0, 6.0};
TEX R0, f[TEX2], TEX1, 2D;
MULR R0.xyz, R0, C1;
TEX R1, f[TEX0], TEX0, 2D;
MULR R1.xyz, R1, f[COL0];
MULR R0.xyz, R0, C6.x;
MULR R0.xyz, R0, R1;
MULR R0.w, R0.w, f[COL0].w;
MOVR o[COLR], R0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
