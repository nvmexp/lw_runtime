vs_1_1
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord2 v2
m4x4 oPos, v0, c[0]
mov oT0, v1
add r5.xyz, c[4].xyz, -v0.xyz
dp3 r5.w, r5.xyz, r5.xyz
rsq r5.w, r5.w
mul r5.xyz, r5.xyz, r5.w
mov oT4.xyz, r5
dp3 r0.x, r5, -c[6]
add r0.x, v0.w, -r0.x
mul r1.xyz, c[7].xyz, r0.x
add oD0, r1.xyz, c[8].xyz
dp4 oT1.x, v0, c[13]
dp4 oT1.y, v0, c[14]
dp4 oT1.z, v0, c[15]
mov oT1.w, v0.w
mov oT2, v2
mov oT3, v0
