!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C3={1, 2, 3, 4};
TEX R2, f[TEX0], TEX0, 2D;
TEXC RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MULR R0.xyz, R2, C3;
MOVR R0.w, C3.w;
MULR R0.xyz, f[COL0], R0;
MULR R0.xyz, C0, R0;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MULR R1, R2, C1;
MADR R0.xyz, R2.w, -R0, R0;
MADR R0.xyz, R2.w, R1, R0;
MOVR o[COLR], R0; 
END

# Passes = 8 

# Registers = 3 

# Textures = 2 
