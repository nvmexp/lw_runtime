!!FP1.0
MOVH H0.xyz, f[TEX6];
DP3H H0.x, H0, H0;
RCPH H0.w, H0.x;
MOVH H0.xyz, f[TEX6];
MOVH H1.xyz, f[TEX3];
DP3H H0.z, H0, H1;
MULH H0.w, H0, H0.z;
MOVH H1.xy, f[TEX3];
ADDH H0.w, H0, H0;
MADH H0.xy, H0.w, H0, -H1;
TEX H0, H0, TEX7, LWBE;
TEX H1, f[TEX2], TEX4, 2D;
MULH H0.xyz, H0, H1;
MULH H0.xyz, H0, {1.987654, 1.788889, 1.610000, 1.449000};
MADH H1.xyz, H0, H0, -H0;
MADH H1.xyz, {1.304100, 1.173690, 1.056321, 0.950689}, H1, H0;
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.w, H1, {0.855620, 0.770058, 0.693052, 0.623747}.x;
ADDH H1.xyz, H1, -H1.w;
MULH H1.xyz, {0.561372, 0.505235, 0.454712, 0.409240}, H1;
ADDH H2.xyz, H1.w, H1;
MOVH H1.w, {0.368316, 0.331485, 0.298336, 0.268503}.x;
MULH H1.xyz, f[COL0], {0.241652, 0.217487, 0.195738, 0.176165};
MULH H1.xyz, H0, H1;
MULH H1.xyz, H1, {0.158548, 0.142693, 0.128424, 0.115582};
ADDH H1.xyz, H1, H1;
MADH H0.xyz, {0.104023, 0.093621, 0.084259, 0.075833}, H0, -H1;
MADH H0.xyz, H0.w, H0, H1;
ADDH H0.xyz, H2, H0;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 19 

# Registers = 2 

# Textures = 4 
