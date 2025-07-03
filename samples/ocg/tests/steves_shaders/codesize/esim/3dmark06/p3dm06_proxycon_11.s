ps_3_0

def c5, 1.00000000, 0.00000000, 2.00000000, -1.00000000 ; 0x3f800000 0x000000 0x40000000 0xbf800000
def c6, 16.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x41800000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord5_pp v2.rgb
dcl_texcoord6 v3.rgb
dcl_texcoord7 v4.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
cmp r0.w, -v3.z, c5.x, c5.y
if_ne r0.w, -r0.w
mov oC0, c5.y
else
dp3_pp r0.w, v2, v2
rsq_pp r0.w, r0.w
nrm_pp r5.xyz, v3
mad_pp r0.xyz, v2, r0.w, r5
dp3 r1.w, v4, v4
nrm_pp r1.xyz, r0
texld r0, v0.xyx, s2
mad_pp r4.xyz, c5.z, r0.wyzw, c5.w
add_sat_pp r4.w, -r1.w, c5.x
dp3_pp r2.x, r4, r1
texld_pp r3, v0.xyx, s1
mul_pp r2.w, r3.w, c4.x
mov r2.yz, c5.y
texldl_pp r1, r2.xwzy, s4
mov r0, c0
mul r0, r0, c1.x
mul_pp r2.xyz, r3, c3
mul_pp r0, r4.w, r0
mul_pp r1, r1.x, r0
mul_pp r2, r2, r1
dp3_pp r1.w, r4, r5
mul_sat_pp r3.w, r1.w, c6.x
mov_sat_pp r3.z, r1.w
texld_pp r1, v0.xyx, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.z
mul_pp r2, r2, r3.w
mul_pp r0, r1, r0
texld_pp r1, v1.xyx, s3
mul_pp r2, r2, r1
mad oC0, r0, r1, r2
endif
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex ffef -texrange 1ff
