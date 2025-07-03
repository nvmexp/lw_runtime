ps_3_0
def c5, 1.00000000, 2.00000000, -1.00000000, 16.00000000 ; 0x3f800000 0x40000000 0xbf800000 0x41800000
def c6, 0.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x000000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2_pp v2.rgb
dcl_texcoord6 v3.rgb
dcl_texcoord7 v4.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dp3_pp r0.w, v2, v2
rsq_pp r0.w, r0.w
nrm_pp r1.xyz, v3
mad_pp r0.xyz, v2, r0.w, r1
nrm_pp r2.xyz, r0
texld r0, v0, s2
mad_pp r0.xyz, c5.y, r0.wyzw, c5.z
dp3_pp r2.x, r0, r2
dp3_pp r2.w, r0, r1
texld_pp r1, v0, s1
mul_pp r2.y, r1.w, c4.x
texld_pp r0, r2, s4
dp3 r0.w, v4, v4
add_sat_pp r0.w, -r0.w, c5.x
mov r2.xyz, c0
mul r2.xyz, r2, c1.x
mul_pp r3.xyz, r0.w, r2
mul_pp r1.xyz, r1, c3
mul_pp r0.xyz, r0.x, r3
mul_pp r0.xyz, r1, r0
mul_sat_pp r0.w, r2.w, c5.w
mov_sat_pp r1.w, r2.w
mul_pp r2.xyz, r0, r0.w
texld_pp r0, v1, s3
mul_pp r3.xyz, r3, r1.w
texld_pp r1, v0, s0
mul_pp r1.xyz, r1, c2
mul_pp r2.xyz, r2, r0
mul_pp r1.xyz, r3, r1
mad oC0.xyz, r1, r0, r2
mov oC0.w, c6.x
