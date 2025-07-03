vs_2_0
def c[1], 765.005859, 1.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v4
mul r0.w, v1.z, c[1].x
mova a0.w, r0.w
dp3 r3.x, v2, c[a0.w+42]
dp3 r3.y, v2, c[a0.w+43]
dp3 r3.z, v2, c[a0.w+44]
dp3 oT3.x, r3, c[8]
dp3 oT3.y, r3, c[9]
dp3 oT3.z, r3, c[10]
mov r0.w, c[1].y
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
dp4 r1.y, r0, c[10]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add oT4.xyz, -r0, c[2]
dp4 oT0.x, v3, c[91]
dp4 oT0.y, v3, c[92]
dp3 r2.x, v4, c[a0.w+42]
dp3 r2.y, v4, c[a0.w+43]
dp3 r2.z, v4, c[a0.w+44]
dp3 r1.x, r2, c[8]
mul oT1.x, r1.x, c[93].x
mul r1.xyz, r3.zxyw, r2.yzxw
mad r1.xyz, r3.yzxw, r2.zxyw, -r1
mov oT7.xyz, r3
mul r1.xyz, r1, v4.w
dp3 r3.x, r1, c[8]
mul oT1.y, r3.x, c[93].x
dp3 r3.x, r2, c[9]
mov oT5.xyz, r2
mul oT2.x, r3.x, -c[93].x
dp3 r2.x, r1, c[9]
mov oT6.xyz, r1
mul oT2.y, r2.x, -c[93].x
dp4 r1.x, r0, c[8]
mov oPos.x, r1.x
dp4 r1.z, r0, c[9]
dp4 r2.x, r0, c[11]
mov oPos.y, r1.z
mul r1.y, r1.z, c[94].w
mov oPos.w, r2.x
add r0.xy, r1, r2.x
mul r0.xy, r0, c[0].w
mov oT1.z, r0.x
mov oT2.z, r0.y
mov oT1.w, r2.x
mov oT2.w, r2.x
