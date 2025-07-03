ps_3_0
def c11, 0.03125000, 0.41608700, -0.30338100, 0.06250000 ; 0x3d000000 0x3ed5095b 0xbe9b54c1 0x3d800000
def c12, 0.13519500, 0.22041900, -0.18368299, 0.07725400 ; 0x3e0a708f 0x3e61b585 0xbe3c1765 0x3d9e3758
def c13, -0.25281799, -0.23776400, 1.00000000, 0.00000000 ; 0xbe81715c 0xbe737868 0x3f800000 0x80000000
def c14, -0.05412700, 0.66291302, -0.03125000, 16.00000000 ; 0xbd5db446 0x3f29b4ab 0xbd000000 0x41800000
def c15, -0.48613599, 0.39774799, -0.39774799, 0.50000000 ; 0xbef8e6d1 0x3ecba5a0 0xbecba5a0 0x3f000000
def c16, 0.57452399, -0.06250000, -0.57452399, 0.10825300 ; 0x3f131401 0xbd800000 0xbf131401 0x3dddb3c0
def c17, -0.62500000, -0.75000000, 0.87500000, 0.18750000 ; 0xbf200000 0xbf400000 0x3f600000 0x3e400000
def c18, 2.00000000, -1.00000000, 0.15000001, 1.00000000 ; 0x40000000 0xbf800000 0x3e19999a 0x3f800000
def c19, 0.50000000, 1.00000000, 0.31830987, 50.00000000 ; 0x3f000000 0x3f800000 0x3ea2f983 0x42480000
dcl_texcoord0 v0.rg
dcl_texcoord2 v1.rgb
dcl_texcoord3 v2.rgb
dcl_texcoord4 v3.rgb
dcl_texcoord5 v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6.rgb
dcl_position1 vPos.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dcl_2d s6
dcl_lwbe s7
dcl_lwbe s8
dsx r0, v6.xyxy
dsy r1, v6.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c3.x
mad_pp r1, r0, r1.w, c2.xyxy
mul r0.xy, vPos, c11.x
texld_pp r0, r0, s1
mul_pp r0, r1, r0.zwxy
mul_pp r1, r0, c11.yyz
mad_pp r1, r0.zwxy, c12.xxy, r1
add r2, r1, v6.xyxy
texldl r1, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r0, c12.zzw
mad_pp r2, r0.zwxy, c13.xxy, r2
mov r1.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r1.z, r3.x
mov r1.w, r2.x
add r2, r1, -v6.z
mul_pp r1, r0, c14.xxy
cmp_pp r2, r2, c13.z, c13.w
mad_pp r1, r0.zwxy, c14.zzy, r1
dp4_pp r5.w, r2, c11.w
add r1, r1, v6.xyxy
texldl r2, r1.xyxy, s0
texldl r3, r1.zwzw, s0
mul_pp r1, r0, c15.xxy
mad_pp r1, r0.zwxy, c15.xxz, r1
mov r2.y, r3.x
add r3, r1, v6.xyxy
texldl r1, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r2.z, r1.x
mul_pp r1, r0, c16.xxy
mov r2.w, r3.x
mad_pp r1, r0.zwxy, c16.zzw, r1
add r2, r2, -v6.z
add r3, r1, v6.xyxy
texldl r1, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r1.y, r3.x
mad r3, r0, c17.xxy, v6.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r1.z, r4.x
mov r1.w, r3.x
cmp_pp r2, r2, c13.z, c13.w
add r1, r1, -v6.z
dp4 r2.w, r2, c11.w
cmp_pp r1, r1, c13.z, c13.w
add_pp r3.w, r5.w, r2.w
dp4 r3.z, r1, c11.w
mad r2, r0, c17.zzw, v6.xyxy
mad r1, r0.zwzw, c19.xxy, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.y, r2.x
texldl r2, r1.xyxy, s0
texldl r1, r1.zwzw, s0
mov r0.z, r2.x
mov r0.w, r1.x
add_pp r2.w, r3.w, r3.z
add r0, r0, -v6.z
cmp_pp r0, r0, c13.z, c13.w
mul r1.xy, v0, c5
texld r1, r1, s5
mad_pp r1.xyz, c18.x, r1.wyzw, c18.y
dp4 r1.w, r0, c11.w
rcp_pp r0.w, r1.z
mul r1.xy, r1, r0.w
mul r0.xy, v0, c4
texld r0, r0, s4
mad_pp r4.xyz, c18.x, r0.wyzw, c18.y
mul r0.xy, r1, c18.z
rcp_pp r0.w, r4.z
mad r0.xy, r4, r0.w, r0
mov_pp r0.z, c18.w
add_pp r0.w, r2.w, r1.w
dp3 r0.z, r0, r0
mov r1.xyz, c0
mul r2.xyz, r1, c1.x
rsq r6.z, r0.z
mul_pp r6.xy, r0, r6.z
nrm_pp r7.xyz, v5
mul_pp r5.xyz, r0.w, r2
dp3_sat_pp r0.w, r6, r7
mul_pp r0.xyz, r5, r0.w
texld r1, v0.xyx, s2
mul_pp r3.xyz, r1, c6
nrm_pp r8.xyz, r4
mul_pp r1.xyz, r0, r3
texld r0, v0.xyx, s6
mul_pp r1.xyz, r1, r0
nrm_pp r4.xyz, v4
dp3_pp r2.w, r8, r4
dp3_pp r0.w, r4, -r7
add_sat_pp r2.w, -r2.w, c18.w
mad_sat_pp r2.z, r0.w, c15.w, c15.w
mul_sat_pp r0.w, r7.z, c14.w
mul_pp r3.w, r2.w, r2.z
mul_pp r1.xyz, r1, r0.w
pow r2.w, r3.w, c9.x
mul_pp r1.xyz, r1, c19.z
mul r2.w, r2.w, c8.x
add_pp r8.xyz, r7, r4
mul r2.w, r2.x, r2.w
mad_pp r4.xyz, r2.w, r1.w, r1
dp3_pp r1.x, r6, v1
dp3_pp r1.y, r6, v2
dp3_pp r1.z, r6, v3
texld_pp r2, r1, s7
texld_pp r1, r1, s8
nrm_pp r7.xyz, r8
dp3_sat_pp r2.w, r6, r7
pow_pp r1.w, r2.w, c19.w
lrp_pp r6.xyz, c10.x, r1, r2
mul_pp r2.xyz, r5, r1.w
texld r1, v0.xyx, s3
mul_pp r5.xyz, r1, c7
mul_pp r1.xyz, r0, r6
mul_pp r2.xyz, r2, r5
mad r1.xyz, r1, r3, r4
mul_pp r0.xyz, r0, r2
mad oC0.xyz, r0, r0.w, r1
mov oC0.w, c13.w
