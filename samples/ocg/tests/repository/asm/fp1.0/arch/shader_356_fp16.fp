!!FP1.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
MOVH H0.xyz, f[TEX1];
MOVH H0.w, C0.x;
DP3H H4.w, f[TEX2], f[TEX2];
LG2H H4.w, |H4.w|;
MULH H4.w, H4, {0.5, 0, 0, 0}.x; 
EX2H H4.w, -H4.w;
MULH H4.xyz, H4.w, f[TEX2];
MOVH H4.w, f[COL0].x;
TEX H6.xyz, f[TEX0], TEX0, 2D;
MOVH H6.w, C0.x;
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 4 

# Textures = 3 
