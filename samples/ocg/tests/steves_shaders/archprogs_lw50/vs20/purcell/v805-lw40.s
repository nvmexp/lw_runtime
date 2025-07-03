vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_blendweight0 v2
mov a0.x, v2.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
dp3 r2.x, c[a0.x+0], v1
dp3 r2.y, c[a0.x+1], v1
dp3 r2.z, c[a0.x+2], v1
mov a0.x, v2.w
dp4 r1.x, c[a0.x+0], v0
dp4 r1.y, c[a0.x+1], v0
dp4 r1.z, c[a0.x+2], v0
dp3 r3.x, c[a0.x+0], v1
dp3 r3.y, c[a0.x+1], v1
dp3 r3.z, c[a0.x+2], v1
mul r0.xyz, r0.xyz, v2.x
mad r0.xyz, r1.xyz, v2.z, r0.xyz
mov r0.w, v0.w
mul r2.xyz, r2.xyz, v2.x
mad r2.xyz, r3.xyz, v2.z, r2.xyz
m4x4 oPos, r0, c[0]
mov oT0.xyz, r2.xyz
add oT1.xyz, c[4].xyz, -r0.xyz
