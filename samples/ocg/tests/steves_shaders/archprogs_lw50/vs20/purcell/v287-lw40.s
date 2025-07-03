vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
mov oT0, v2
add r2.xyz, c[4], -v0
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
dp3 r2.x, r2.xyz, v1.xyz
max r2.x, r2.x, c[4].w
mul r2, r2.x, c[5]
add oD0, r2, c[6]
