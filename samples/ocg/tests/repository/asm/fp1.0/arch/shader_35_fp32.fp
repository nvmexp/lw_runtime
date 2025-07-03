!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX0, 2D;
TEXC RC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
MULR R2.xyz, R0, C2;
MOVR R2.w, C2.w;
MULR R2.xyz, f[COL0], R2;
MULR R2.xyz, C0, R2;
MULR R2.xyz, R2, {2, 0, 0, 0}.x; 
MULR R3, R0, C1;
MADR R4.xyz, R0.w, -R2, R2;
MADR H0.xyz, R0.w, R3, R4;
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 8 

# Registers = 5 

# Textures = 2 
