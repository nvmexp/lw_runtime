vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v3
dp3 r1.x, v1, c[42]
dp3 r1.y, v1, c[43]
dp3 r1.z, v1, c[44]
dp3 oT6.x, r1, c[8]
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp3 oT6.y, r1, c[9]
dp4 r2.z, r0, c[9]
dp3 oT6.z, r1, c[10]
mov r3.y, -r2.z
mov oPos.y, r2.z
dp4 r3.x, r0, c[8]
dp4 r4.y, r0, c[11]
dp4 r5.y, r0, c[10]
add r2.xyz, -r0, c[2]
add r0.xy, r3, r4.y
mov oPos.x, r3.x
mul oT5.xy, r0, c[0].w
mad oFog, -r5.y, c[16].w, c[16].x
dp3 r3.x, v3, c[42]
dp3 r3.y, v3, c[43]
dp3 r3.z, v3, c[44]
mov oPos.z, r5.y
mul r0.xyz, r1.zxyw, r3.yzxw
dp3 oT1.x, r2, r3
mad r0.xyz, r1.yzxw, r3.zxyw, -r0
mov oT2.x, r3.x
mov oT3.x, r3.y
mov oT4.x, r3.z
mul r0.xyz, r0, v3.w
dp3 oT1.y, r2, r0
dp3 oT1.z, r2, r1
mov oT2.z, r1.x
mov oT3.z, r1.y
mov oT4.z, r1.z
dp4 oT0.x, v2, c[91]
dp4 oT0.y, v2, c[92]
mov oPos.w, r4.y
mov oT5.z, r4.y
mov oT2.y, r0.x
mov oT3.y, r0.y
mov oT4.y, r0.z
