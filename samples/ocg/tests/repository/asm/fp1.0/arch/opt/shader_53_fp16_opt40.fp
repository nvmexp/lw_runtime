!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX H0.xyz, f[TEX0], TEX0, 2D;
MOVH H2, f[COL0];
MULH H2.xyz, H0, H2;
TEX H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
MULH_m2 H2.xyz, C0, H2;
MADH H2.xyz, H2.w, -H2, H2;
MULH H1.xy, C1, H0;
MULH H1.w, C1.xywz, H0.xywz;
MADH H0.xyz, H2.w, H1.xywz, H2;
MOVH H0.w, H2.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
