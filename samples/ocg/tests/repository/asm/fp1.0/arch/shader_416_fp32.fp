!!FP1.0
MOVR R1.w, f[TEX2];
MOVR R1.xyz, f[TEX1];
MOVR R0.w, f[TEX1];
RCPR R0.w, R0.w;
MULR R1.xyz, R0.w, R1;
MOVR R2.yzw, f[TEX2].wxyz;
TEX R0, f[TEX0], TEX0, 2D;
DP3R R2.x, R1, R0;
RCPR R0.w, R1.w;
MULR R1.xyz, R0.w, R2.yzwy;
MOVR R1.w, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3R R2.y, R1, R0;
TEX R0, R2, TEX2, 2D;
MULR H0.xyz, R0, {1.304100, 1.173690, 1.056321, 0.950689}.x;
MOVR H0.w, R1;
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 3 

# Textures = 3 
