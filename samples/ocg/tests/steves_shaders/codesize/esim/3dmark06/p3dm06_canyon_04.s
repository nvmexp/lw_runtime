ps_3_0

def c4, 0.66666669, -1.00000000, 2.00000000, 0.33333334 ; 0x3f2aaaab 0xbf800000 0x40000000 0x3eaaaaab
def c5, 0.20000000, 0.03125000, 0.66291302, 0.57452399 ; 0x3e4ccccd 0x3d000000 0x3f29b4ab 0x3f131401
def c6, 0.87500000, -0.57452399, 1.00000000, 0.00000000 ; 0x3f600000 0xbf131401 0x3f800000 0x80000000
def c7, 0.06250000, -0.75000000, -0.48613599, -0.39774799 ; 0x3d800000 0xbf400000 0xbef8e6d1 0xbecba5a0
def c8, -0.62500000, 0.39774799, 0.50000000, 0.41608700 ; 0xbf200000 0x3ecba5a0 0x3f000000 0x3ed5095b
def c9, 0.13519500, 0.22041900, -0.25281799, 0.07725400 ; 0x3e0a708f 0x3e61b585 0xbe81715c 0x3d9e3758
def c10, -0.30338100, -0.18368299, -0.23776400, 0.18750000 ; 0xbe9b54c1 0xbe3c1765 0xbe737868 0x3e400000
def c11, 0.10825300, -0.03125000, -0.06250000, -0.05412700 ; 0x3dddb3c0 0xbd000000 0xbd800000 0xbd5db446
def c12, 16.00000000, 0.31830987, 0.00000000, 0.00000000 ; 0x41800000 0x3ea2f983 0x000000 0x000000
dcl_texcoord0 v0
dcl_color0 v1
dcl_color1 v2
dcl_texcoord4 v3.rg
dcl_texcoord1 v4.rgb
dcl_texcoord2 v5.rgb
dcl_texcoord3 v6.rgb
dcl_texcoord6 v7.rgb
dcl_texcoord7 v8.rgb
dcl vPos.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dcl_2d s6
dcl_2d s7
dcl_2d s8
dcl_2d s9
dcl_2d s10
dcl_lwbe s11
dsx r0, v8.xyxy
dsy r1, v8.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c2.x
mad_pp r1, r0, r1.w, c1.xyxy
mul r0.xy, vPos, c5.y
texld_pp r0, r0, s1
mul_pp r2, r1, r0.zwxy
mul_pp r0.zw, r2, c5.z
mad_pp r0.zw, r2.xyxy, c5.z, r0
mov_pp r0.xy, r2.zwzw
add r0, r0, v8.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c6.xxy
mad_pp r0.zw, r2.xyxy, c5.w, r0
mov r1.y, r3.x
add r0, r0, v8.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2.zwxy, c7.yyz
mad_pp r0.zw, r2, c7.z, r0
add r1, r1, -v8.z
add r3, r0, v8.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r2, c8.xxy
mad_pp r3.zw, r2.xyxy, c7.w, r3
mov r0.y, r4.x
add r3, r3, v8.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
cmp_pp r1, r1, c6.z, c6.w
add r0, r0, -v8.z
dp4_pp r1.w, r1, c7.x
cmp_pp r0, r0, c6.z, c6.w
dp4 r1.z, r0, c7.x
mul_pp r0, r2.zwxy, c8.zzw
mad_pp r0.zw, r2, c9.x, r0
add_pp r5.w, r1.w, r1.z
add r0, r0, v8.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c9.yyz
mad_pp r0, r2.zwxy, c10.xxy, r0
mov r1.y, r3.x
add r0, r0, v8.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2, c10.zzw
mad_pp r0.xy, r2.zwzw, c9.w, r0
add r1, r1, -v8.z
add r3, r0, v8.xyxy
texldl r0, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mul_pp r4, r2, c11.xxy
mad_pp r2, r2.zwxy, c11.zzw, r4
mov r0.y, r3.x
add r2, r2, v8.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
cmp_pp r1, r1, c6.z, c6.w
add r0, r0, -v8.z
dp4 r1.w, r1, c7.x
cmp_pp r0, r0, c6.z, c6.w
add_pp r2.w, r5.w, r1.w
dp4 r2.z, r0, c7.x
texld r0, v1.zwz, s5
texld r1, v2.xyx, s6
add r1.xyz, r0.wyzw, r1.wyzw
texld r0, v2.zwz, s7
add r0.xyz, r1, r0.wyzw
add_pp r1.w, r2.w, r2.z
mad_pp r4.xyz, r0, c4.x, c4.y
texld r0, v2.zwz, s9
texld r2, v1.xyx, s8
texld r3, v3.xyx, s10
add_pp r0.x, r2.w, r3.w
add_pp r0.x, r0.x, c4.y
mad_pp r1.xyz, c4.z, r0.wyzw, c4.y
mul_sat_pp r3.z, r0.x, c3.x
mul_pp r0, r1.w, c0
lrp_pp r5.xyz, r3.z, r1, r4
texld r1, v0.xyx, s2
texld r4, v0.zwz, s3
add r1, r1, r4
texld r4, v1.xyx, s4
add r1, r1, r4
nrm_pp r6.xyz, v7
mul_pp r4, r1, c4.w
dp3_sat_pp r3.w, r5, r6
lrp_pp r1, r3.z, r2, r4
mul_pp r0, r0, r3.w
mul_pp r1, r1, c5.x
mul_pp r0, r0, r1
mul_sat_pp r2.w, r6.z, c12.x
mul_pp r0, r3.x, r0
mul_pp r0, r2.w, r0
mul_pp r2, r0, c12.y
dp3_pp r0.x, r5, v4
dp3_pp r0.y, r5, v5
dp3_pp r0.z, r5, v6
texld_pp r0, r0, s11
mul_pp r0, r3.x, r0
mad oC0, r0, r1, r2
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex f7fc -texrange 7ffff8
