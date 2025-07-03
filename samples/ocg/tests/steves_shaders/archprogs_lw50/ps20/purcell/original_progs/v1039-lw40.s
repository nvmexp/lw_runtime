vs_2_0
def c[1], 765.005859, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xy, v2.zyzw, c[1].x
mova a0.xy, r0
mul r0, v1.y, c[a0.y+42]
mad r0, c[a0.x+42], v1.x, r0
dp3 r4.x, v6, r0
mul r2, v1.y, c[a0.y+43]
mul r1, v1.y, c[a0.y+44]
mad r2, c[a0.x+43], v1.x, r2
mad r1, c[a0.x+44], v1.x, r1
dp3 r4.y, v6, r2
dp3 r4.z, v6, r1
dp3 r3.x, v3, r0
dp4 r0.x, v0, r0
dp3 r3.y, v3, r2
dp3 r3.z, v3, r1
dp4 r0.y, v0, r2
mul r2.xyz, r4.yzxw, r3.zxyw
dp4 r0.z, v0, r1
mad r1.xyz, r3.yzxw, r4.zxyw, -r2
mov oT4.xyz, r4
mov r0.w, c[0].y
mul oT5.xyz, r1, v6.w
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
add oT3.xyz, -r0, c[2]
mad oFog, -r1.y, c[16].w, c[16].x
slt r0.xyz, r3, c[0].x
mova a0.xyz, r0
mul r0.xyz, r3, r3
mov oPos.z, r1.y
mul r1.xyz, r0.y, c[a0.y+23]
mov oT6.xyz, r3
mad r1.xyz, r0.x, c[a0.x+21], r1
mad oD0.xyz, r0.z, c[a0.z+25], r1
mul r0.xy, v4, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v4, c[91]
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
