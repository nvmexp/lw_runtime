vs_2_0
def c[1], 765.005859, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mul r0.w, v1.z, c[1].x
mova a0.w, r0.w
dp3 r2.x, v5, c[a0.w+42]
dp3 r2.y, v5, c[a0.w+43]
dp3 r2.z, v5, c[a0.w+44]
dp3 r1.x, v2, c[a0.w+42]
dp3 r1.y, v2, c[a0.w+43]
dp3 r1.z, v2, c[a0.w+44]
mul r0.xyz, r2.yzxw, r1.zxyw
mad r0.xyz, r1.yzxw, r2.zxyw, -r0
mov oT4.xyz, r2
mul oT5.xyz, r0, v5.w
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
mov r0.w, c[0].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r3.y, r0, c[10]
add oT3.xyz, -r0, c[2]
slt r0.xyz, r1, c[0].x
mova a0.xyz, r0
mul r0.xyz, r1, r1
mad oFog, -r3.y, c[16].w, c[16].x
mul r2.xyz, r0.y, c[a0.y+23]
mov oPos.z, r3.y
mad r2.xyz, r0.x, c[a0.x+21], r2
mad r0.xyz, r0.z, c[a0.z+25], r2
dp3 r2.x, r1, -c[28]
mov oT6.xyz, r1
max r0.w, r2.x, c[0].x
mad oD0.xyz, c[27], r0.w, r0
mul r0.xy, v3, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v3, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oD0.w, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
