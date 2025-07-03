#include "ReShade.fxh"
#include "ReShadeUI.fxh"

uniform float3 float_3 <
	ui_type = "drag";
	ui_label = "Grouped Slider Float3 Top Left";
	ui_tooltip = "Grouped Slider Float3 Hint.";
	ui_min = float3(-1.7, -1.8, -1.9); ui_max = float3(1.4, 1.5, 1.6);
	ui_step = float3(0.1, 0.2, 0.3);
	> = float3(0.4, 0.5, 0.6);

uniform int2 int_2 <
	ui_type = "drag";
	ui_label = "Grouped Slider Int2 Top Middle";
	ui_tooltip = "Grouped Slider Int2 Hint.";
	ui_min = int2(-10, -20); ui_max = int2(11, 22);
	ui_step = int2(1, 2);
	> = int2(3, 4);

uniform float4 color_rgba <
	ui_type = "color";
	ui_label = "Color Picker Top Right";
	ui_tooltip = "Color Picker Hint.";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform float3 edit_box <
	ui_type = "input";
	ui_label = "Grouped Input Box Float3 Bottom Left";
	ui_tooltip = "Grouped Input Box Float3 hint.";
	> = float3(0.0, 0.0, 0.0);

uniform int dropdown_box_5 <
	ui_label = "Dropdown Box 5 Bottom Middle Left";
	ui_tooltip = "Dropdown Box Hint.";
	ui_items = "Option 0 Red\0Option 1 Green\0Option 2 Blue\0Option 3 Yellow\0Option 4 Cyan\0";
> = 3;

uniform int radio_button_5 <
	ui_type = "radio";
	ui_label = "Radio Button 5 Bottom Middle Right";
	ui_tooltip = "Radio Button Hint.";
	ui_items = "ROption 0 Red\0ROption 1 Green\0ROption 2 Blue\0ROption 3 Yellow\0ROption 4 Cyan\0";
> = 2;

uniform float regular_float <
	ui_type = "drag";
	ui_min = -1.0; ui_max = 1.0;
    ui_step = 0.05;
    ui_label = "Regular Float Bottom Right";
	ui_tooltip = "Regular float hint.";
> = 0.5;

uniform bool checkbox
<
    ui_label = "Checkbox Ilwert All";
	ui_tooltip = "Checkbox hint.";
> = false;

float3 PostProcessPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
	float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
	
	if (texcoord.x >= 0 && texcoord.x < 0.33
		&& texcoord.y >= 0 && texcoord.y < 0.5)
	{
		color.r = color.r * float_3.x;
		color.g = color.g * float_3.y;
		color.b = color.b * float_3.z;
	}
	else if (texcoord.x >= 0.33 && texcoord.x < 0.67
		&& texcoord.y >= 0 && texcoord.y < 0.5)
	{
		color.r = color.r * ((float)(int_2.x)/10.0);
		color.g = color.g * ((float)(int_2.y)/10.0);
	}
	else if (texcoord.x >= 0.67 && texcoord.x < 1.0
		&& texcoord.y >= 0 && texcoord.y < 0.5)
	{
		color.r = color.r * color_rgba.x;
		color.g = color.g * color_rgba.y;
		color.b = color.b * color_rgba.z;
		
		color.r = color.r * color_rgba.w;
		color.g = color.g * color_rgba.w;
		color.b = color.b * color_rgba.w;
	}
	else if (texcoord.x >= 0 && texcoord.x < 0.25
		&& texcoord.y >= 0.5 && texcoord.y < 1.0)
	{
		color.r = color.r * edit_box.x;
		color.g = color.g * edit_box.y;
		color.b = color.b * edit_box.z;
	}
	else if (texcoord.x >= 0.25 && texcoord.x < 0.50
		&& texcoord.y >= 0.5 && texcoord.y < 1.0)
	{
		float3 colorChange = float3(1.0,1.0,1.0);
		if		(dropdown_box_5 == 0) colorChange = float3(1.0,0.0,0.0);
		else if (dropdown_box_5 == 1) colorChange = float3(0.0,1.0,0.0);
		else if (dropdown_box_5 == 2) colorChange = float3(0.0,0.0,1.0);
		else if (dropdown_box_5 == 3) colorChange = float3(1.0,1.0,0.0);
		else if (dropdown_box_5 == 4) colorChange = float3(0.0,1.0,1.0);
		
		color.r = color.r * colorChange.x;
		color.g = color.g * colorChange.y;
		color.b = color.b * colorChange.z;
	}
	else if (texcoord.x >= 0.50 && texcoord.x < 0.75
		&& texcoord.y >= 0.5 && texcoord.y < 1.0)
	{
		float3 colorChange = float3(1.0,1.0,1.0);
		if		(radio_button_5 == 0) colorChange = float3(1.0,0.0,0.0);
		else if (radio_button_5 == 1) colorChange = float3(0.0,1.0,0.0);
		else if (radio_button_5 == 2) colorChange = float3(0.0,0.0,1.0);
		else if (radio_button_5 == 3) colorChange = float3(1.0,1.0,0.0);
		else if (radio_button_5 == 4) colorChange = float3(0.0,1.0,1.0);
		
		color.r = color.r * colorChange.x;
		color.g = color.g * colorChange.y;
		color.b = color.b * colorChange.z;
	}
	else if (texcoord.x >= 0.75 && texcoord.x < 1.0
		&& texcoord.y >= 0.5 && texcoord.y < 1.0)
	{
		color.r = color.r * regular_float;
		color.g = color.g * regular_float;
		color.b = color.b * regular_float;
	}
	
	if (checkbox)
	{
		color.r = 1.0 - color.r;
		color.g = 1.0 - color.g;
		color.b = 1.0 - color.b;
	}
	
	return color;
}

technique Test
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = PostProcessPS;
	}
}
