!!FP1.0
DECLARE C0={0.1, 0.3, 0.5, 0.9};
DECLARE C1={0.25, 0.22, 0.27, 0.33};
TEX R1, f[TEX1], TEX0, 2D;
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, C0.w, R0.x;
MOVR R0.x, f[TEX0];
MADR R0.w, C0.y, R1.x, R0;
TEX R1, f[TEX2], TEX0, 2D;
MADR R0.w, C0.y, R1.x, R0;
TEX R1, f[TEX3], TEX0, 2D;
MADR R0.w, C1.w, R1.x, R0;
MULR R0.w, C0.z, R0;
MADR R0.w, C1.w, R0.x, R0;
MADR R0.w, R0, C0.z, C0.x;
FRCR R0.w, R0;
MADR R0.w, R0, C1.w, C1.x;
MULR R0.w, R0, R0;
MADR R0.z, R0.w, C0.z, C0.x;
MADR R0.z, R0.w, R0, C1.y;
MADR R0.z, R0.w, R0, C1.x;
MADR R0.z, R0.w, R0, C1.z;
MADR R0.w, R0, R0.z, C0.x;
TEX R0, R0.w, TEX2, 2D;
MOVR o[COLR], R0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
