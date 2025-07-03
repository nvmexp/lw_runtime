!!FP2.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
DP3H H4.w, f[TEX2], f[TEX2];
MOVH H0.w, C0.x;
LG2H_d2 H4.w, |H4.w|;
MOVH H0.xyz, f[TEX1];
EX2H H4.w, -H4.w;
MOVH H6.w, C0.x;
MULH H4.xyz, H4.w, f[TEX2];
TEX H6.xyz, f[TEX0], TEX0, 2D;
MOVH H4.w, f[COL0].x;
END

# Passes = 8 

# Registers = 4 

# Textures = 3 
