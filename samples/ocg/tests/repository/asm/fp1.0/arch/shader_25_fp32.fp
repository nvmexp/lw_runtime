!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C3={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
TEXC RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MULR R0, R0, C3;
MULR R0.xyz, f[COL0], R0;
MULR H0.xyz, C0, R0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 2 
