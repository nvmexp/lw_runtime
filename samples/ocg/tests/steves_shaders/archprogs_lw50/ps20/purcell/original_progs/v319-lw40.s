vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_texcoord1 v3
dcl_texcoord2 v4
dcl_tangent0 v5
m4x4 oPos, v0, c[0]
mov oT0, v2
mov oT1, v3
mov oT2, v4
mov oT3, v3
mov r5, v5
mul r1.xyz, v1.yzx, r5.zxy
mad r1.xyz, -r5.yzx, v1.zxy, r1.xyz
dp3 oT4.x, c[4], v5
dp3 oT4.y, c[4], r1
dp3 oT4.z, c[4], v1
mov oT4.w, v0.w
mov oD0, c[5]
dp4 oT5.x, v0, c[6]
dp4 oT5.y, v0, c[8]
