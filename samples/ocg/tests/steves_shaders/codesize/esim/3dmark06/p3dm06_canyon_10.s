ps_3_0

def c9, 0.15000001, 0.03125000, 0.66291302, 0.57452399 ; 0x3e19999a 0x3d000000 0x3f29b4ab 0x3f131401
def c10, 0.87500000, -0.57452399, 1.00000000, 0.00000000 ; 0x3f600000 0xbf131401 0x3f800000 0x80000000
def c11, 0.06250000, -0.75000000, -0.48613599, -0.39774799 ; 0x3d800000 0xbf400000 0xbef8e6d1 0xbecba5a0
def c12, -0.62500000, 0.39774799, 0.50000000, 0.41608700 ; 0xbf200000 0x3ecba5a0 0x3f000000 0x3ed5095b
def c13, 0.13519500, 0.22041900, -0.25281799, 0.07725400 ; 0x3e0a708f 0x3e61b585 0xbe81715c 0x3d9e3758
def c14, -0.30338100, -0.18368299, -0.23776400, 0.18750000 ; 0xbe9b54c1 0xbe3c1765 0xbe737868 0x3e400000
def c15, 0.10825300, -0.03125000, -0.06250000, -0.05412700 ; 0x3dddb3c0 0xbd000000 0xbd800000 0xbd5db446
def c16, 2.00000000, -1.00000000, 0.20000000, 1.00000000 ; 0x40000000 0xbf800000 0x3e4ccccd 0x3f800000
def c17, 0.10000000, 1.00000000, 16.00000000, 0.31830987 ; 0x3dcccccd 0x3f800000 0x41800000 0x3ea2f983
def c18, -0.89999998, 10.00000000, -2.00000000, 3.00000000 ; 0xbf666666 0x41200000 0xc0000000 0x40400000
def c19, 0.15000001, 0.30000001, 0.40000001, 1.00000000 ; 0x3e19999a 0x3e99999a 0x3ecccccd 0x3f800000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rgb
dcl_texcoord2 v2.rgb
dcl_texcoord3 v3.rgb
dcl_texcoord4 v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6.rgb
dcl vPos.rg
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
mov r1.w, c2.x
mad_pp r1, r0, r1.w, c1.xyxy
mul r0.xy, vPos, c9.y
texld_pp r0, r0, s1
mul_pp r1, r1, r0.zwxy
mul_pp r0.zw, r1, c9.z
mad_pp r0.zw, r1.xyxy, c9.z, r0
mov_pp r0.xy, r1.zwzw
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r1, c10.xxy
mad_pp r2.zw, r1.xyxy, c9.w, r2
mov r0.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
add r0, r0, -v6.z
cmp_pp r2, r0, c10.z, c10.w
mul_pp r0, r1.zwxy, c11.yyz
mad_pp r0.zw, r1, c11.z, r0
dp4_pp r5.w, r2, c11.x
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r1, c12.xxy
mad_pp r2.zw, r1.xyxy, c11.w, r2
mov r0.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
add r2, r0, -v6.z
mul_pp r0, r1.zwxy, c12.zzw
mad_pp r0.zw, r1, c13.x, r0
cmp_pp r2, r2, c10.z, c10.w
add r3, r0, v6.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r1, c13.yyz
mad_pp r3, r1.zwxy, c14.xxy, r3
mov r0.y, r4.x
add r3, r3, v6.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
dp4 r2.w, r2, c11.x
add r0, r0, -v6.z
add_pp r2.w, r5.w, r2.w
cmp_pp r0, r0, c10.z, c10.w
dp4 r2.z, r0, c11.x
mul_pp r0, r1, c14.zzw
mad_pp r0.xy, r1.zwzw, c13.w, r0
add_pp r4.w, r2.w, r2.z
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mul_pp r3, r1, c15.xxy
mad_pp r1, r1.zwxy, c15.zzw, r3
mov r0.y, r2.x
add r1, r1, v6.xyxy
texldl r2, r1.xyxy, s0
texldl r1, r1.zwzw, s0
mov r0.z, r2.x
mul r2.xy, v0, c3
texld r2, r2, s4
mad_pp r5.xyz, c16.x, r2.wyzw, c16.y
nrm r8.xyz, v1
mov r0.w, r1.x
dp3_pp r1.w, -r8, r5
add r0, r0, -v6.z
add_pp r1.w, r1.w, r1.w
mad_pp r1.xyz, r5, -r1.w, -r8
nrm_pp r9.xyz, v5
cmp_pp r0, r0, c10.z, c10.w
dp3 r1.w, r9, r1
dp4 r0.z, r0, c11.x
add r0.w, r1.w, c18.x
add_pp r0.y, r4.w, r0.z
mul_sat r0.z, r0.w, c18.y
mad r0.w, r0.z, c18.z, c18.w
mul r0.z, r0.z, r0.z
mul_pp r4, r0.y, c0
mul r0.w, r0.w, r0.z
mul r1, r4, r0.w
texld r0, v0.xyx, s3
mul_pp r0, r0, c6
dp3_pp r3.w, r5, r8
mul r2, r1, r0
add_sat_pp r6.y, -r3.w, c16.w
mul_pp r0, r0, r0
mul_pp r1.w, r6.y, r6.y
mad_sat_pp r5.w, r1.w, r1.w, c9.x
dp3_pp r1.w, r5, r9
mul r2, r2, r5.w
mov_sat_pp r6.xzw, r1.w
texldl r1, r6, s6
mul r3.xy, v0, c4
texld r3, r3, s5
mad_pp r3.xyz, c16.x, r3.wyzw, c16.y
rcp_pp r3.w, r3.z
mul r3.xy, r3, r3.w
mul r3.xy, r3, c16.z
rcp_pp r3.w, r5.z
mad r3.xy, r5, r3.w, r3
mov_pp r3.z, c16.w
dp3 r3.w, r3, r3
rsq r7.z, r3.w
mul_pp r7.xy, r3, r7.z
dp3_sat_pp r3.w, r7, r9
add_pp r6.w, -r5.w, c16.w
mul_pp r1, r1, r3.w
mul r5, r2, c19
mul_pp r1, r4, r1
texld r2, v0.xyx, s2
mul_pp r3, r2, c5
mul_pp r1, r1, r3
dp3_sat_pp r2.z, r8, -r9
mad_pp r2.z, r2.z, r6.y, -c12.z
mad_pp r2.y, r2.w, -c17.x, c17.y
mul_pp r1, r1, c17.w
mad_pp r2.z, r2.z, r2.y, c12.z
mad_pp r1, r1, r6.w, r5
mul_pp r2.xyz, r4, r2.z
mul_sat_pp r4.w, r9.z, c17.z
mul_pp r2.xyz, r2, c7
mul_pp r1, r1, r4.w
mul_pp r2.xyz, r2, c8.x
mad_pp r1.xyz, r2, r2.w, r1
dp3_pp r2.x, r7, v2
dp3_pp r2.y, r7, v3
dp3_pp r2.z, r7, v4
texld_pp r2, r2, s7
dp3 r4.w, -r8, r7
add r4.w, r4.w, r4.w
mul_pp r2, r3, r2
mad r4.xyz, r7, -r4.w, -r8
dp3_sat r4.w, r7, r8
dp3_pp r3.x, r4, v2
dp3_pp r3.y, r4, v3
dp3_pp r3.z, r4, v4
texld_pp r3, r3, s8
add r4.w, -r4.w, c16.w
mul r4.w, r4.w, r4.w
mad_pp r0, r0, r3, -r2
mad_sat_pp r3.w, r4.w, r4.w, c9.x
mad_pp r0, r3.w, r0, r2
add oC0, r1, r0
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex fe7c -texrange 17ff8 -partialtexld 4000
