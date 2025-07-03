!!FP2.0
DECLARE C0={0.1, 0.3, 0.5, 0.9};
DECLARE C1={0.25, 0.22, 0.27, 0.33};
TEX H1, f[TEX1], TEX1, 2D;
MADH H1.w, H1, C0.z, C0.y;
MULH H1.w, H1, C0.w;
TEX H0, f[TEX0], TEX0, 2D;
EX2H H1.w, H1.w;
MULH H2.xyz, H1, H1.w;
MADH H0.w, H0, C0.z, C0.y;
MULH H0.w, H0, C0.w;
EX2H H0.w, H0.w;
MULH H1.xyz, H0, H0.w;
TEX H0, f[TEX1], TEX2, 2D;
MADH H0.w, H0, C0.w, C0.x;
MULH H1.xyz, H1, C0.w;
MULH H0.w, H0, C0.z;
MADH H1.xyz, H2, C1, H1;
EX2H H0.w, H0.w;
MULH H0.xyz, H0, H0.w;
MADH H0.xyz, H0, C1, H1;
MOVH H1.xy, f[TEX2];
DP2AH H0.w, C1.x, H1, H1.xyxy;
ADDH H0.w, -H0, C0.w;
MULH H1.w, H0, H0;
MULH H0.w, H0, H1;
MULH H0.xyz, H0, H0.w;
MOVH H0.w, C0.z;
LG2H H0.x, |H0.x|;
LG2H H0.z, |H0.z|;
LG2H H0.y, |H0.y|;
MULH H0.xyz, H0, C0.w;
EX2H H0.x, H0.x;
EX2H H0.z, H0.z;
EX2H H0.y, H0.y;
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
