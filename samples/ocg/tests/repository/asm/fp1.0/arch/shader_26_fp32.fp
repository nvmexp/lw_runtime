!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R1, f[TEX0], TEX0, 2D;
TEX R2, f[TEX1], TEX1, 2D;
TEXC RC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
ADDR R0.w, {1, 1, 1, 1}, -R2;
MULR R0.xyz, R1, R2;
MULR H0.xyz, C0, R0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 3 

# Textures = 3 
