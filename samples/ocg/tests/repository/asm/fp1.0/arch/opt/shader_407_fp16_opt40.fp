!!FP2.0
DECLARE C5={1.000000, 0.333333, 2.000000, 0.000000};
MOVH H0.xyz, f[TEX6];
DP3H H0.w, H0, H0;
RCPH H0.w, H0.w;
MOVH H1.xyz, f[TEX3];
DP3H H0.z, H0, H1;
MULH H0.w, H0, H0.z;
MOVH H2.xy, f[TEX3];
ADDH H0.w, H0, H0;
TEX H1, f[TEX0], TEX0, 2D;
ADDH H1.w, -H1, C5.x;
MADH H0.xy, H0.w, H0, -H2;
TEX H0, H0, TEX1, 2D;
MULH H0.xyz, H0, H1.w;
MOVH H0.w, C5.x;
MULH H0.xyz, H0, C5.x;
MADH H2.xyz, H0, H0, -H0;
MADH H0.xyz, C5.x, H2, H0;
MULH H2.xyz, f[COL0], C5.x;
DP3H H1.w, H0, C5.x;
ADDH H0.xyz, H0, -H1.w;
MULH H0.xyz, C5.x, H0;
ADDH H0.xyz, H1.w, H0;
MULH H1.xyz, H1, H2;
MULH H1.xyz, H1, C5.x;
MADH H0.xyz, C5.x, H1, H0;
END

# Passes = 14 

# Registers = 2 

# Textures = 3 
