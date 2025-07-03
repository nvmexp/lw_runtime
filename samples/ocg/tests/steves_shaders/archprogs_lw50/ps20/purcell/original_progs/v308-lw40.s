vs_1_1
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord2 v2
dcl_blendweight0 v3
mov a0.x, v3.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
mul r4, c[a0.x+0], v3.x
mul r5, c[a0.x+1], v3.x
mul r6, c[a0.x+2], v3.x
mov a0.x, v3.w
dp4 r1.x, c[a0.x+0], v0
dp4 r1.y, c[a0.x+1], v0
dp4 r1.z, c[a0.x+2], v0
mul r0.xyz, r0.xyz, v3.x
mad r0.xyz, r1.xyz, v3.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mov oT0, v1
mov oT1, v2
mad oT2, c[a0.x+0], v3.z, r4
mad oT3, c[a0.x+1], v3.z, r5
mad oT4, c[a0.x+2], v3.z, r6
dp4 r0.x, r0, c[3]
add r0.x, r0.x, -c[4].x
mul oT5, r0.x, c[4].y
