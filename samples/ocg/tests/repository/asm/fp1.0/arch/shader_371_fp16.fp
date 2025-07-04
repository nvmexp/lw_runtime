!!FP1.0
DECLARE C0={ 0.2, 0.1, 0.3, 0.2};
DECLARE C1={ 0.5, 0.4, 0.3, 0.1};
DECLARE C2={-0.004000, 1.000000, 0.150000, 0.000000};
DECLARE C3={1.000000, -1.000000, 0.254545, 256.000000};
DECLARE C4={0.500000, -0.500000, 0.000000, -2.000000};
DECLARE C5={2.000000, -1.000000, 0.000000, 0.000000};
MOVH H0.w, f[TEX1];
RCPH H0.w, H0.w;
MOVH H1.xy, f[TEX1];
MULH H0.xy, H0.w, H1;
ADDH H0.xy, C3, H0;
MOVH H2.xyz, f[TEX4];
MULH H0.xy, H0, C4;
ADDH H0.xy, H0, C0;
TEX H1, H0, TEX4, 2D;
MULH H1.xyz, C5.x, H1;
DP3H H1.w, H2, H2;
LG2H H0.z, |H1.w|;
MULH H0.z, H0, {0.5, 0, 0, 0}.x; 
ADDH H3.xyz, C5.y, H1;
EX2H H0.w, -H0.z;
MOVH H1.xyz, f[TEX4];
MULH H2.xyz, H0.w, H1;
MOVH H1.xyz, f[TEX3];
DP3H H0.w, H1, H1;
LG2H H0.z, |H0.w|;
MULH H0.z, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.w, -H0.z;
MOVH H1.xyz, f[TEX3];
MULH H1.xyz, H0.w, H1;
DP3H H3.w, H3, H1;
MULH H1.w, H3, C4.w;
MULH H3.xyz, H3, H1.w;
ADDH H1.xyz, H1, H3;
DP3H H0.w, H1, -H2;
MOVH H2.xyz, f[TEX2];
SGEH H0.z, H0.w, {0, 0, 0, 0}.x;
ADDH H0.w, H0, -C2.w;
MULH H0.w, H0.z, H0;
TEX H1, H0, TEX1, 2D;
ADDH H0.w, C2.w, H0;
MULH H1.w, H1, C3.w;
LG2H H0.w, |H0.w|;
MULH H1.w, H0, H1;
EX2H H1.w, H1.w;
MULH H1.xyz, H1, H1.w;
TEX H0, H0, TEX0, 2D;
ADDH H0.w, H3, -C2.w;
SGEH H1.w, H3, {0, 0, 0, 0}.x;
MULH H1.w, H1, H0;
ADDH H0.w, C2.w, H1;
MULH H0.xyz, H0, H0.w;
DP3H H1.w, H2, H2;
ADDH H3, C2.w, -C2.w;
ADDH H0.xyz, H1, H0;
MULH H1.xyz, H0, C1;
LG2H H0.x, |H1.w|;
MULH H0.x, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.w, -H0.x;
MULH H0.w, H0, H1;
ADDH H2.w, H0, C2.x;
TEX H0, f[TEX2], TEX7, LWBE;
SGEH H0, H0, {0, 0, 0, 0};
MULH H0, H0, H3;
ADDH H0, C2.y, H0;
DP4H H0.w, H0, C2.z;
ADDH H0.w, -H0, C2.y;
MULH H1.xyz, H1, H0.w;
SGEH H0.w, H1, -{0, 0, 0, 0}.x;
ADDH H1.w, -H1, C2.y;
ADDH H0.z, H1.w, -C2.w;
MULH H0.w, H0, H0.z;
ADDH H1.w, C2.w, H0;
MULH H1.w, H1, H1;
MOVH H0.w, C2.y;
MULH H0.xyz, H1, H1.w;
LG2H H0.x, |H0.x|;
LG2H H0.y, |H0.y|;
LG2H H0.z, |H0.z|;
MULH H0.xyz, H0, C3.z;
EX2H H0.x, H0.x;
EX2H H0.y, H0.y;
EX2H H0.z, H0.z;
MOVH o[COLH], H0;
END

# Passes = 51 

# Registers = 4 

# Textures = 4 
