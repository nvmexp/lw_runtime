ps_3_0

def c8, 0.50000000, 27.80170441, 0.15915494, 1.00000000 ; 0x3f000000 0x41de69e4 0x3e22f983 0x3f800000
def c9, 6.28318548, -3.14159274, 0.00100000, 0.03503015 ; 0x40c90fdb 0xc0490fdb 0x3a83126f 0x3d0f7bc6
def c10, 0.99681526, -0.07974523, 0.00000000, 59.27533340 ; 0x3f7f2f49 0xbda35177 0x000000 0x426d19f1
def c11, 0.99920094, 0.03996804, 0.00000000, 0.15000001 ; 0x3f7fcba2 0x3d23b587 0x000000 0x3e19999a
def c12, 1.00086141, 1.85840690, 2.20579910, 1.12623131 ; 0x3f801c3a 0x3fede047 0x400d2bd0 0x3f902859
def c13, 0.99755895, 0.06982913, 0.00000000, 88.49556732 ; 0x3f7f6006 0x3d8f0293 0x000000 0x42b0fdbb
def c14, 0.03125000, 0.66291302, 0.87500000, -0.57452399 ; 0x3d000000 0x3f29b4ab 0x3f600000 0xbf131401
def c15, -0.75000000, -0.48613599, -0.62500000, 0.39774799 ; 0xbf400000 0xbef8e6d1 0xbf200000 0x3ecba5a0
def c16, 0.50000000, 0.41608700, 0.13519500, 0.07725400 ; 0x3f000000 0x3ed5095b 0x3e0a708f 0x3d9e3758
def c17, 0.22041900, -0.25281799, -0.30338100, -0.18368299 ; 0x3e61b585 0xbe81715c 0xbe9b54c1 0xbe3c1765
def c18, -0.23776400, 0.18750000, 0.10825300, -0.03125000 ; 0xbe737868 0x3e400000 0x3dddb3c0 0xbd000000
def c19, -0.06250000, -0.05412700, 16.00000000, 0.00059683 ; 0xbd800000 0xbd5db446 0x41800000 0x3a1c74a7
def c20, 1.00000000, 0.00000000, 0.06250000, -0.48613599 ; 0x3f800000 0x000000 0x3d800000 0xbef8e6d1
def c21, 0.00100000, 0.02798440, -0.00223875, 0.01250000 ; 0x3a83126f 0x3ce53f8a 0xbb12b806 0x3c4ccccd
def c22, 0.00100000, 0.04086730, 0.00163469, 0.20000000 ; 0x3a83126f 0x3d276478 0x3ad64329 0x3e4ccccd
def c23, 0.31830987, 0.00000000, 0.00000000, 0.00000000 ; 0x3ea2f983 0x000000 0x000000 0x000000
def c24, 0.00100000, 0.05031934, 0.00352235, 66.84239960 ; 0x3a83126f 0x3d4e1ba7 0x3b66d74b 0x4285af4f
dcl_texcoord0 v0
dcl_texcoord1 v1
dcl_texcoord2 v2.rgb
dcl_texcoord3 v3.rg
dcl_texcoord4 v4.rga
dcl_texcoord7 v5.rgb
dcl vPos.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
dcl_2d s6
dcl_2d s7
dcl_2d s8
dcl_2d s9
dcl_2d s10
dcl_2d s11
dsx r0, v5.xyxy
dsy r1, v5.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c3.x
mad_pp r1, r0, r1.w, c2.xyxy
mul r0.xy, vPos, c14.x
texld_pp r0, r0, s1
mul_pp r2, r1, r0.zwxy
mul_pp r0.zw, r2, c14.y
mad_pp r0.zw, r2.xyxy, c14.y, r0
mov_pp r0.xy, r2.zwzw
add r0, r0, v5.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c14.zzw
mad_pp r0.zw, r2.xyxy, -c14.w, r0
mov r1.y, r3.x
add r0, r0, v5.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2.zwxy, c15.xxy
mad_pp r0.zw, r2, c20.w, r0
add r1, r1, -v5.z
add r3, r0, v5.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r2, c15.zzw
mad_pp r3.zw, r2.xyxy, -c15.w, r3
mov r0.y, r4.x
add r3, r3, v5.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
cmp_pp r1, r1, c20.x, c20.y
add r0, r0, -v5.z
dp4_pp r1.w, r1, c20.z
cmp_pp r0, r0, c20.x, c20.y
dp4 r1.z, r0, c20.z
mul_pp r0, r2.zwxy, c16.xxy
mad_pp r0.zw, r2, c16.z, r0
add_pp r5.w, r1.w, r1.z
add r0, r0, v5.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c17.xxy
mad_pp r0, r2.zwxy, c17.zzw, r0
mov r1.y, r3.x
add r0, r0, v5.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2, c18.xxy
mad_pp r0.xy, r2.zwzw, c16.w, r0
add r1, r1, -v5.z
add r3, r0, v5.xyxy
texldl r0, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mul_pp r4, r2, c18.zzw
mad_pp r2, r2.zwxy, c19.xxy, r4
mov r0.y, r3.x
add r2, r2, v5.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
cmp_pp r1, r1, c20.x, c20.y
add r0, r0, -v5.z
dp4 r1.w, r1, c20.z
cmp_pp r0, r0, c20.x, c20.y
add_pp r1.w, r5.w, r1.w
dp4 r0.w, r0, c20.z
add_pp r0.w, r1.w, r0.w
mul_pp r2, r0.w, c0
mul r3.xyz, v0.xyxw, c8.xxyw
mov r0.w, c7.x
mul r0, r0.w, c12
add r0.x, r3.z, r0.x
dp2add r1.w, c13, r3, c13.z
mad r0.x, r0.x, c8.z, c8.x
mad r0.y, r1.w, c13.w, r0.y
frc r0.x, r0.x
mad r0.y, r0.y, c8.z, c8.x
mad r0.x, r0.x, c9.x, c9.y
frc r0.y, r0.y
sincos r1.xy, r0.x
mad r0.y, r0.y, c9.x, c9.y
mul r5.xy, r1.yxzw, c9.zwzw
sincos r1.xy, r0.y
add_pp r0.y, -r5.x, c8.w
mul r4.xyz, r1.yxxw, c24
add_pp r1.w, r0.y, -r4.x
dp2add r0.x, c10, r3, c10.z
dp2add r0.y, c11, r3, c11.z
mad r0.z, r0.x, c24.w, r0.z
mad r0.w, r0.y, c10.w, r0.w
mad r0.z, r0.z, c8.z, c8.x
frc r0.z, r0.z
mad r0.w, r0.w, c8.z, c8.x
mad r1.y, r0.z, c9.x, c9.y
frc r1.z, r0.w
sincos r0.xy, r1.y
mad r1.z, r1.z, c9.x, c9.y
mul r3.xyz, r0.yxxw, c21
sincos r0.xy, r1.z
add_pp r0.w, r1.w, -r3.x
mul r1.xyz, r0.yxxw, c22
add_pp r0.y, r0.w, -r1.x
add_pp r0.z, -r5.y, -r4.y
add_pp r0.w, -r4.z, -r3.z
add_pp r0.z, -r3.y, r0.z
add_pp r0.x, -r1.y, r0.z
add_pp r0.z, -r1.z, r0.w
texld r1, v0.zwzw, s5
rcp r0.w, r1.y
nrm_pp r3.xyz, r0
mul r4.xy, r1.xzzw, r0.w
texld r0, v0, s5
rcp r1.w, r0.y
rcp_pp r0.w, r3.y
mad r4.xy, r0.xzzw, r1.w, r4
mad r4.xz, r3, r0.w, r4.xyyw
mov r4.y, c8.w
dp3 r0.w, r1, r1
dp3 r1.z, r4, r4
dp3 r1.w, r0, r0
rsq r0.y, r1.z
mul r0.xz, r4, r0.y
rsq r1.w, r1.w
rcp r1.w, r1.w
rsq r0.w, r0.w
mul r0.xyz, r0, r1.w
rcp r1.w, r0.w
dp3_pp r0.w, r3, r3
mul r0.xyz, r0, r1.w
rsq_pp r0.w, r0.w
nrm r4.xyz, v2
rcp_pp r0.w, r0.w
add_pp r1.xyz, r4, c1
mul_pp r6.xyz, r0, r0.w
nrm_pp r0.xyz, r1
dp3_pp r1.y, r6, r6
dp3_pp r1.x, r6, r0
texld_pp r1, r1, s8
dp3_pp r3.w, r6, c1
mov_sat_pp r0.w, r3.w
mul_pp r0, r2, r0.w
mul_pp r1, r2, r1.x
mul_pp r2, r0, c5
mul_sat_pp r4.w, r3.w, c19.z
mul_pp r3, r2, c19.w
dp3 r2.w, r6, r4
mad_pp r1, r1, r4.w, -r3
texld_pp r5, r2.w, s6
texld r2, v1, s2
mul_pp r4, r2, c22.w
mad_pp r3, r5, r1, r3
mul_pp r0, r0, r4
mad_pp r1, r0, c23.x, -r3
texld_pp r0, v1.zwzw, s3
add_pp r9.w, r2.w, r0.x
mul_pp r2, r6.xzxz, c6.xyxy
texld_pp r0, r6, s4
rcp r6.w, v4.w
mul r6.xyz, r2.xzw, r6.w
mul r8.xy, r6.w, v4
add r2.x, r6.x, r8.x
add r7.xy, r6.yzzw, r8
add r2.y, r2.y, r8.y
texld_pp r6, r2, s11
texld r2, r7, s9
texld_pp r8, r7, s10
add r2.w, r2.x, -v3.x
mul_sat r2.w, r2.w, v3.y
texld_pp r2, r2.w, s7
mov_pp r7, c5
mad_pp r7, r7, c21.w, -r8
mad_pp r7, r2.x, r7, r8
mul_pp r2, r0, c5
mad_pp r2, r2, c21.w, -r7
mad_pp r7, r2, c11.w, r7
add_pp r8.w, r9.w, -c8.w
lrp_pp r2, r5, r6, r7
mul_sat_pp r5.w, r8.w, c4.x
mad_pp r0, r0, r4, -r2
mad_pp r1, r5.w, r1, r3
mad_pp r0, r5.w, r0, r2
add oC0, r1, r0
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex f08c -texrange 51d9f8 -partialtexld 100
