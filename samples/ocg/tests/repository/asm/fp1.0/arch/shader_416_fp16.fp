!!FP1.0
MOVH H1.w, f[TEX2];
MOVH H1.xyz, f[TEX1];
MOVH H0.w, f[TEX1];
RCPH H0.w, H0.w;
MULH H1.xyz, H0.w, H1;
MOVH H2.yzw, f[TEX2].wxyz;
TEX H0, f[TEX0], TEX0, 2D;
DP3H H2.x, H1, H0;
RCPH H0.w, H1.w;
MULH H1.xyz, H0.w, H2.yzwy;
MOVH H1.w, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3H H2.y, H1, H0;
TEX H0, H2, TEX2, 2D;
MULH H0.xyz, H0, {1.304100, 1.173690, 1.056321, 0.950689}.x;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 8 

# Registers = 2 

# Textures = 3 
