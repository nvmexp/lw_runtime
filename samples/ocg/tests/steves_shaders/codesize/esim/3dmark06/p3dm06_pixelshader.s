ps_3_0

def c10, 2.00000000, -1.00000000, 0.25000000, 16.00000000 ; 0x40000000 0xbf800000 0x3e800000 0x41800000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord2_pp v2.rgb
dcl_texcoord3_pp v3.rgb
dcl_texcoord4 v4.rgb
dcl_texcoord7 v5.rgb
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
dcl_lwbe s10
dcl_2d s11
mul r0.xy, v0, c2
texld r0, r0, s0
mul r1.xy, v0, c3
texld r1, r1, s1
add r0, r0, r1
mul r1.xy, v0, c4
texld r1, r1, s2
add r0, r0, r1
mul r1.xy, v0, c5
texld r1, r1, s3
mul r2.xy, v0, c6
texld r2, r2, s4
mad_pp r3.xyz, c10.x, r2.wyzw, c10.y
mul r2.xy, v0, c7
texld r2, r2, s5
mad_pp r2.xyz, c10.x, r2.wyzw, r3
add_pp r3.xyz, r2, c10.y
mul r2.xy, v0, c8
texld r2, r2, s6
mad_pp r2.xyz, c10.x, r2.wyzw, r3
add_pp r3.xyz, r2, c10.y
mul r2.xy, v0, c9
texld r2, r2, s7
mad_pp r2.xyz, c10.x, r2.wyzw, r3
add_pp r2.xyz, r2, c10.y
add r0, r0, r1
mul_pp r6.xyz, r2, c10.z
mul_pp r2, r0, c10.z
dp3_pp r0.x, r6, v1
dp3_pp r0.y, r6, v2
dp3_pp r0.z, r6, v3
texld_pp r0, r0, s10
texld_pp r1, v0, s9
mul_pp r0, r0, r1.x
mul_pp r0, r2, r0
nrm_pp r3.xyz, v5
dp3_sat_pp r1.z, r6, r3
add_sat_pp r1.w, r3.z, r3.z
mul_pp r1.z, r1.z, r1.w
nrm_pp r7.xyz, v4
mov r4.xyz, c0
mul_pp r5.xyz, r4, c1.x
add_pp r8.xyz, r3, r7
mul_pp r4.xyz, r1.z, r5
nrm_pp r3.xyz, r8
dp3_pp r8.x, r6, r3
dp3_pp r8.y, r6, r6
texld_pp r3, r8, s11
rsq_pp r1.z, r8.y
mul_pp r6.xyz, r6, r1.z
dp3_sat_pp r1.z, r6, r7
mul_pp r1.w, r1.w, r3.x
add_pp r1.z, -r1.z, -c10.y
mul_pp r2.xyz, r2, r4
mul_pp r1.w, r1.w, r1.z
mul_sat_pp r4.xyz, r1.x, r2
mul_pp r3.xyz, r5, r1.w
texld r2, v0, s8
mul_pp r5.xyz, r2, c10.w
add r2.xyz, r0, r4
mul_pp r3.xyz, r3, r5
mad r2.xyz, r4, -r0, r2
mul_sat_pp r0.xyz, r1.x, r3
add r1.xyz, r2, r0
mov_pp oC0.w, r0.w
mad oC0.xyz, r0, -r2, r1
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex f7ff -texrange 7fffff
