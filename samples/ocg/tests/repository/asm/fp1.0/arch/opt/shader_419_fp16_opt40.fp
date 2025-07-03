!!FP2.0
TEX H0, f[TEX0], TEX3, 2D;
NRMH H1.xyz, f[TEX1];
DP3H H0.z, H1, H0;
MOVH H1.xy, f[TEX2];
ADDH H0.z, -H0, {0.855620, 0.770058, 0.693052, 0.623747}.x;
MULH H1.w, H0.z, H0.z;
DP2AH H2.w, H1.x, f[TEX3], H0.xyxy;
MOVH H1.xz, f[TEX4].xxyy;
MULH H1.w, H1, H1;
MOVH H2.xy, f[TEX7];
MULH H1.w, H0.z, H1;
MOVH H0.z, f[TEX2];
DP2AH H3.w, H1.y, H1.xzxx, H0.xyxy;
RCPH H1.z, H0.z;
MULH H3.x, H2.w, H1.z;
MOVH H1.xy, f[TEX5];
MULH H3.y, H1.z, H3.w;
TEX H3, H3, TEX4, 2D;
DP2AH H2.y, H1.y, H2, {0.987, 0, 0, 0}.x;
DP2AH H2.x, H1.x, f[TEX6], H0.xyxy;
MULH H0.xy, H1.z, H2.xyyy;
TEX H2, H0, TEX2, 2D;
MULH H0.xyz, H3, {0.561372, 0.505235, 0.454712, 0.409240};
MULH H1.xyz, H2, {0.368316, 0.331485, 0.298336, 0.268503};
MADH H0.xyz, H0, H1.w, H1;
END

# Passes = 13 

# Registers = 2 

# Textures = 8 
