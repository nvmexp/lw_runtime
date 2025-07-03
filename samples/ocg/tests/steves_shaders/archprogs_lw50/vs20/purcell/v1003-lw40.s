vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
mov oT0, v2
dp3 oT1.x, v1, c[4]
dp3 oT1.y, v1, c[5]
dp3 oT1.z, v1, c[6]
add r2.xyz, c[7].xyz, -v0.xyz
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
dp3 r3.x, r2, v1
add r3.x, r3.x, r3.x
mad r2.xyz, r3.x, v1.xyz, -r2.xyz
dp3 oT2.x, r2, c[4]
dp3 oT2.y, r2, c[5]
dp3 oT2.z, r2, c[6]
mov oT3, v2
