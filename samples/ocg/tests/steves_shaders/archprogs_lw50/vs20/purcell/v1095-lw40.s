vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xyz, v2.zyxw, c[1].x
mova a0.xyz, r0
add r1.w, -v1.x, c[0].y
mul r0, v1.y, c[a0.y+42]
add r4.w, r1.w, -v1.y
mad r0, c[a0.x+42], v1.x, r0
mad r3, c[a0.z+42], r4.w, r0
dp4 r0.x, v0, r3
mul r1, v1.y, c[a0.y+43]
mad r2, c[a0.x+43], v1.x, r1
mul r1, v1.y, c[a0.y+44]
mad r2, c[a0.z+43], r4.w, r2
mad r1, c[a0.x+44], v1.x, r1
dp4 r0.y, v0, r2
mad r1, c[a0.z+44], r4.w, r1
dp4 r0.z, v0, r1
mov r0.w, c[0].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp3 r4.x, v6, r3
dp3 r3.x, v3, r3
dp3 r4.y, v6, r2
dp3 r3.y, v3, r2
dp3 r4.z, v6, r1
dp3 r3.z, v3, r1
dp4 r8.y, r0, c[10]
mul r1.xyz, r4.yzxw, r3.zxyw
add oT3.xyz, -r0, c[2]
mad r0.xyz, r3.yzxw, r4.zxyw, -r1
dp3 r2.x, c[1].yzzw, r4
mul r0.xyz, r0, v6.w
dp3 r2.z, c[1].yzzw, r3
dp3 r2.y, c[1].yzzw, r0
slt r1.xyz, r2, c[0].x
mul r5.xyz, r2, r2
mova a0.xyw, r1.xyz
mul r1.xy, r4.yzzw, c[3]
mul r2.xyz, r5.y, c[a0.y+23]
add r1.x, r1.y, r1.x
mul r7.xy, r3.yzzw, c[3]
mul r6.xy, r0.yzzw, c[3]
add r1.z, r7.y, r7.x
add r1.y, r6.y, r6.x
mad r6.xyz, r5.x, c[a0.x+21], r2
slt r2.xyz, r1, c[0].x
mul r1.xyz, r1, r1
mova a0.xyz, r2
mad oFog, -r8.y, c[16].w, c[16].x
mul r2.xyz, r1.y, c[a0.y+23]
mov oPos.z, r8.y
mad r2.xyz, r1.x, c[a0.x+21], r2
mad oD0.xyz, r5.z, c[a0.w+25], r6
mad oD1.xyz, r1.z, c[a0.z+25], r2
dp3 r1.x, c[1].w, r4
mov oT4.x, r4.x
mov oT5.x, r4.y
mov oT6.x, r4.z
dp3 r1.z, c[1].w, r3
dp3 r1.y, c[1].w, r0
mov oT4.z, r3.x
slt r2.xyz, r1, c[0].x
mul r1.xyz, r1, r1
mova a0.xyz, r2
mov oT5.z, r3.y
mul r2.xyz, r1.y, c[a0.y+23]
mov oT6.z, r3.z
mad r2.xyz, r1.x, c[a0.x+21], r2
mad oT7.xyz, r1.z, c[a0.z+25], r2
mul r1.xy, v4, c[90]
add r0.w, r1.y, r1.x
mul r1.xy, v4, c[91]
mov oT0.x, r0.w
add r1.z, r1.y, r1.x
mov oT0.y, r1.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r1.z
mov oT2.y, r1.z
mov oT4.y, r0.x
mov oT5.y, r0.y
mov oT6.y, r0.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
