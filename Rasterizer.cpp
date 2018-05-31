#include "Scene.hpp"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "glad.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>


inline RGB8 toRGB8(vec3 color)
{
	return {
		unsigned char(min(color.x, 1.f) * 255),
		unsigned char(min(color.y, 1.f) * 255),
		unsigned char(min(color.z, 1.f) * 255) };
}

struct Img
{
	RGB8* buf;
	ivec2 size;
};

void setPx(Img img, ivec2 pos, vec3 color)
{
	assert(img.size.x > pos.x);
	assert(img.size.y > pos.y);
	img.buf[img.size.x * pos.y + pos.x] = toRGB8(color);
}

void drawLine(Img img, vec2 startf, vec2 endf, vec3 color)
{
	ivec2 start = ivec2(startf) + 0.5f;
	ivec2 end = ivec2(endf) + 0.5f;

	// clip
	start.x = max(0, start.x);
	start.x = min(img.size.x - 1, start.x);
	start.y = max(0, start.y);
	start.y = min(img.size.y - 1, start.y);

	end.x = max(0, end.x);
	end.x = min(img.size.x - 1, end.x);
	end.y = max(0, end.y);
	end.y = min(img.size.y - 1, end.y);

	bool steep = false;
	if (fabs(start.x - end.x) < fabs(start.y - end.y))
	{
		steep = true;
		swapcpy(start.x, start.y);
		swapcpy(end.x, end.y);
	}

	if (start.x > end.x)
		swapcpy(start, end);

	for (int x = start.x; x < end.x; ++x)
	{
		const float t = float(x - start.x) / (end.x - start.x);
		const int y = start.y + t * (end.y - start.y);

		if (steep)
			setPx(img, { y, x }, color);
		else
			setPx(img, { x, y }, color);

	}
}

void loadModel(Model& model, const char* const filename)
{
	FILE* file = fopen(filename, "r");
	if (!file)
	{
		printf("could not load model: %s\n", filename);
		return;
	}

	char buf[256];
	int pos = 0;
	while (true)
	{
		int c = getc(file);

		if (c == EOF)
			break;

		if (ferror(file))
		{
			printf("loadModel file IO error: %s\n", filename);
			break;
		}

		if (pos == getSize(buf))
		{
			printf("loadModel error (too long line): %s\n", filename);
			break;
		}

		buf[pos] = c;
		++pos;

		if (c == '\n')
		{
			buf[pos] = '\0';
			pos = 0;

			char* start = buf + 2;

			if (buf[0] == 'v')
			{
				vec3 pos = {}; // without {} msvc outputs error
				assert(sscanf(start, "%f%f%f", &pos.x, &pos.y, &pos.z) == 3);
				model.positions.pushBack(pos);
			}
			else if (strncmp(buf, "vt", 2) == 0)
			{
				vec2 texCoord;
				assert(sscanf(start, "%f%f", &texCoord.x, &texCoord.y) == 2);
				model.texCoords.pushBack(texCoord);
			}
			else if (strncmp(buf, "vn", 2) == 0)
			{
				vec3 normal;
				assert(sscanf(start, "%f%f%f", &normal.x, &normal.y, &normal.z) == 3);
				model.normals.pushBack(normal);
			}
			else if (buf[0] == 'f')
			{
				Face face;
				assert(sscanf(start, "%d/%d/%d %d/%d/%d %d/%d/%d",
					&face.indices[0].position, &face.indices[0].texCoord, &face.indices[0].normal,
					&face.indices[1].position, &face.indices[1].texCoord, &face.indices[1].normal,
					&face.indices[2].position, &face.indices[2].texCoord, &face.indices[2].normal ) == 9);

				// in wavefront indicies start from 1
				for (Index& i : face.indices)
				{
					--i.position;
					--i.texCoord;
					--i.normal;
				}
				model.faces.pushBack(face);
			}
		}
	}

	fclose(file);
}

Rasterizer::Rasterizer()
{
	glBuffers_ = createGLBuffers();
	glGenTextures(1, &glTexture_);
	glBindTexture(GL_TEXTURE_2D, glTexture_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

	loadModel(model_, "res/diablo3_pose/diablo3_pose.obj");
}

Rasterizer::~Rasterizer()
{
	deleteGLBuffers(glBuffers_);
	glDeleteTextures(1, &glTexture_);
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
	framebuffer_.resize(frame_.fbSize.x * frame_.fbSize.y);

	for (RGB8& px : framebuffer_)
		px = { 0, 0, 0 };

	Img img = { framebuffer_.data(), ivec2(frame_.fbSize) };

	// rendering
	// @@@@@
	for (const Face& face : model_.faces)
	{
		for (int i = 0; i < 3; ++i)
		{
			vec3 v1 = model_.positions[face.indices[i].position];
			vec3 v2 = model_.positions[face.indices[(i + 1) % 3].position];
			v1.y *= -1.f;
			v2.y *= -1.f;

			const vec2 start = (vec2(v1.x, v1.y) + 1.f) * (frame_.fbSize - 1.f) / 2.f;
			const vec2 end = (vec2(v2.x, v2.y) + 1.f) * (frame_.fbSize - 1.f) / 2.f;

			drawLine(img, start, end, { 1.f, 0.f, 1.f });
		}
	}
	// @@@@@

	bindProgram(program);
	uniform1i(program, "mode", FragmentMode::Texture);
	uniform2f(program, "cameraPos", 0.f, 0.f);
	uniform2f(program, "cameraSize", frame_.fbSize);

	glBindTexture(GL_TEXTURE_2D, glTexture_);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, frame_.fbSize.x, frame_.fbSize.y, 0, GL_RGB, GL_UNSIGNED_BYTE,
		framebuffer_.data());

	Rect rect;
	rect.pos = { 0.f, 0.f };
	rect.size = frame_.fbSize;

	updateGLBuffers(glBuffers_, &rect, 1);
	renderGLBuffers(glBuffers_, 1);

	ImGui::Begin("info");
	ImGui::Text("press Esc to exit");
	ImGui::Text("triangles: %d", model_.faces.size());
	ImGui::End();
}
