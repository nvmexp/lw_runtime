TEMP R0, R1, R2
PARAM defaultTexCoord = { 0, 0.5, 0, 1 }
SUB  R0, program.elw[4], vertex.position
DP3  result.texcoord[0].x, vertex.attrib[9], R0
DP3  result.texcoord[0].y, vertex.attrib[10], R0
DP3  result.texcoord[0].z, vertex.attrib[11], R0
MOV  result.texcoord[1], defaultTexCoord
DP4  result.texcoord[1].x, vertex.attrib[8], program.elw[10]
DP4  result.texcoord[1].y, vertex.attrib[8], program.elw[11]
MOV  result.texcoord[2], defaultTexCoord
DP4  result.texcoord[2].x, vertex.position, program.elw[9]
DP4  result.texcoord[3].x, vertex.position, program.elw[6]
DP4  result.texcoord[3].y, vertex.position, program.elw[7]
DP4  result.texcoord[3].w, vertex.position, program.elw[8]
MOV  result.texcoord[4], defaultTexCoord
DP4  result.texcoord[4].x, vertex.attrib[8], program.elw[12]
DP4  result.texcoord[4].y, vertex.attrib[8], program.elw[13]
MOV  result.texcoord[5], defaultTexCoord
DP4  result.texcoord[5].x, vertex.attrib[8], program.elw[14]
DP4  result.texcoord[5].y, vertex.attrib[8], program.elw[15]
SUB  R0, program.elw[4], vertex.position
DP3  R1, R0, R0
RSQ  R1, R1.x
MUL  R0, R0, R1.x
SUB  R1, program.elw[5], vertex.position
DP3  R2, R1, R1
RSQ  R2, R2.x
MUL  R1, R1, R2.x
ADD  R0, R0, R1
DP3  result.texcoord[6].x, vertex.attrib[9], R0
DP3  result.texcoord[6].y, vertex.attrib[10], R0
DP3  result.texcoord[6].z, vertex.attrib[11], R0
SWZ  result.color, R0, 1, 1, 1, 1
