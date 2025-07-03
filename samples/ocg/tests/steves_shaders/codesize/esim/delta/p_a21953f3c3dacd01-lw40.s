ps_3_0
def c5, 2.00000000, -1.00000000, 16.00000000, 0.00000000 ; 0x40000000 0xbf800000 0x41800000 0x80000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dp3_pp r0.w, v1, v1
rsq_pp r0.w, r0.w
nrm_pp r1.xyz, v2
mad_pp r0.xyz, v1, r0.w, r1
nrm_pp r2.xyz, r0
texld r0, v0, s3
mad_pp r0.xyz, c5.x, r0.wyzw, c5.y
dp3_pp r2.x, r0, r2
dp3_pp r3.w, r0, r1
texld_pp r1, v0, s2
mul_pp r2.y, r1.w, c4.x
texld_pp r0, r2, s4
texldp_pp r2, v3, s0
mov r3.xyz, c0
mul r3.xyz, r3, c1.x
mul_pp r3.xyz, r2.x, r3
mul_pp r1.xyz, r1, c3
mul_pp r0.xyz, r0.x, r3
mul_pp r0.xyz, r1, r0
mul_sat_pp r0.w, r3.w, c5.z
mov_sat_pp r1.w, r3.w
mul_pp r2.xyz, r0, r0.w
texld_pp r0, v0, s1
mul_pp r1.xyz, r0, c2
mul_pp r0.xyz, r3, r1.w
mad oC0.xyz, r0, r1, r2
mov oC0.w, c5.w
