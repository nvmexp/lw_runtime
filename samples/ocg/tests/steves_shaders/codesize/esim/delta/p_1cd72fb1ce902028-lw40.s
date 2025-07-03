ps_3_0
def c5, 2.00000000, -1.00000000, 16.00000000, 0.00000000 ; 0x40000000 0xbf800000 0x41800000 0x80000000
dcl_texcoord0 v0.rg
dcl_texcoord2_pp v1.rgb
dcl_texcoord3_pp v2.rgb
dcl_texcoord4_pp v3.rgb
dcl_texcoord5_pp v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
texld r0, v0.xyx, s3
mad_pp r2.xyz, c5.x, r0.wyzw, c5.y
dp3_pp r0.x, r2, v1
dp3_pp r0.y, r2, v2
dp3_pp r0.z, r2, v3
texld_pp r0, r0, s4
texld_pp r1, v0.xyx, s1
mul_pp r4.xyz, r1, c2
texldp_pp r1, v6, s0
mov r3.xyz, c0
mul r5.xyz, r3, c1.x
nrm_pp r3.xyz, v5
mul_pp r6.xyz, r1.x, r5
dp3_pp r0.w, r2, r3
mov_sat_pp r1.z, r0.w
dp3_pp r1.w, v4, v4
mul_sat_pp r0.w, r0.w, c5.z
rsq_pp r1.w, r1.w
mul_pp r1.xyz, r6, r1.z
mad_pp r3.xyz, v4, r1.w, r3
mul_pp r5.xyz, r4, r1
nrm_pp r1.xyz, r3
dp3_pp r1.x, r2, r1
mov r1.zw, c5.w
texld_pp r2, v0.xyx, s2
mul_pp r1.y, r2.w, c4.x
mul_pp r3.xyz, r2, c3
texldl_pp r1, r1, s5
mul_pp r2.xyz, r6, r1.x
mad r1.xyz, r0, r4, r5
mul_pp r0.xyz, r3, r2
mad oC0.xyz, r0, r0.w, r1
mov oC0.w, c5.w
