#include "Scene.hpp"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"

Rasterizer::Rasterizer()
{

}

Rasterizer::~Rasterizer()
{

}

void Rasterizer::processInput(const Array<WinEvent>& events)
{
	for(const WinEvent& e : events)
	{
		if (e.type == WinEvent::Type::Key && e.key.key == GLFW_KEY_ESCAPE && e.key.action == GLFW_PRESS)
			frame_.popMe = true;
	}
}

void Rasterizer::render(GLuint program)
{
	ImGui::Begin("info");
	ImGui::Text("press Esc to exit");
	ImGui::End();
}
