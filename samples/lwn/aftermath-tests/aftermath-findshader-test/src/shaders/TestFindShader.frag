#version 440 core

layout(binding=0) uniform sampler2D tex2D;
layout(location = 0) out vec4 color;

in IO { vec2 ftc; };

void main()
{
  color = texture(tex2D, ftc);
}

