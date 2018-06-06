#include "Scene.hpp"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "glad.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>

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
		const int c = getc(file);

		if (c == EOF)
			break;

		if (pos == getSize(buf))
		{
			printf("loadModel error (too long line): %s\n", filename);
			break;
		}

		buf[pos] = c;

		if (c != '\n')
		{
			++pos;
			continue;
		}

		buf[pos] = '\0';
		pos = 0;

		char* start = buf + 2;

		if (strncmp(buf, "vt", 2) == 0)
		{
			vec2 texCoord;
			assert(sscanf(start, "%f %f", &texCoord.x, &texCoord.y) == 2);
			model.texCoords.pushBack(texCoord);
		}
		else if (strncmp(buf, "vn", 2) == 0)
		{
			vec3 normal;
			assert(sscanf(start, "%f %f %f", &normal.x, &normal.y, &normal.z) == 3);
			model.normals.pushBack(normal);
		}
		// must be third (so normals and texCoords won't be added here)
		else if (buf[0] == 'v')
		{
			vec3 pos;
			assert(sscanf(start, "%f %f %f", &pos.x, &pos.y, &pos.z) == 3);
			model.positions.pushBack(pos);
		}
		// the same as in 'v'
		else if (buf[0] == 'f')
		{
			Face face;
			assert(sscanf(start, "%d/%d/%d %d/%d/%d %d/%d/%d",
				&face.indices[0].position, &face.indices[0].texCoord, &face.indices[0].normal,
				&face.indices[1].position, &face.indices[1].texCoord, &face.indices[1].normal,
				&face.indices[2].position, &face.indices[2].texCoord, &face.indices[2].normal) == 9);

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

	fclose(file);
}

inline RGB8 toRGB8(vec3 color)
{
	return {
		unsigned char(min(color.x, 1.f) * 255),
		unsigned char(min(color.y, 1.f) * 255),
		unsigned char(min(color.z, 1.f) * 255) };
}

void setPx(Framebuffer& fb, ivec2 pos, vec3 color)
{
	assert(fb.size.x > pos.x);
	assert(fb.size.y > pos.y);
	fb.colorArray[fb.size.x * pos.y + pos.x] = toRGB8(color);
}

// returns false if the depth value was not written
bool setPxDepth(Framebuffer& fb, ivec2 pos, float depth)
{
	assert(fb.size.x > pos.x);
	assert(fb.size.y > pos.y);
	const int idx = fb.size.x * pos.y + pos.x;

	if (fb.depthArray[idx] > depth)
	{
		fb.depthArray[idx] = depth;
		return true;
	}

	return false;
}

void drawLine(Framebuffer& fb, vec3 startf, vec3 endf, vec3 color)
{
	ivec2 start = ivec2(startf.x, startf.y) + 0.5f;
	ivec2 end = ivec2(endf.x, endf.y) + 0.5f;

	// clip
	start.x = max(0, start.x);
	start.x = min(fb.size.x - 1, start.x);
	start.y = max(0, start.y);
	start.y = min(fb.size.y - 1, start.y);
	end.x = max(0, end.x);
	end.x = min(fb.size.x - 1, end.x);
	end.y = max(0, end.y);
	end.y = min(fb.size.y - 1, end.y);

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
			setPx(fb, { y, x }, color);
		else
			setPx(fb, { x, y }, color);

	}
}

vec3 cross(vec3 v, vec3 w)
{
	return  {
		v.y * w.z - v.z * w.y,
		v.z * w.x - v.x * w.z,
		v.x * w.y - v.y * w.x
	};
}

inline float length(vec2 v)
{
	return sqrtf(v.x * v.x + v.y * v.y);
}

inline float length(vec3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline vec2 normalize(vec2 v)
{
	return v * (1.f / length(v));
}

inline vec3 normalize(vec3 v)
{
	return v * (1.f / length(v));
}

inline float dot(vec3 v1, vec3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline float dot(vec2 v1, vec2 v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}

// this is the implementation from tinyrenderer on github
// todo: study the implementation for scratchpixel
vec3 getBarycentric(ivec2 A, ivec2 B, ivec2 C, ivec2 P)
{
	vec3 u = cross(vec3(B.x - A.x, C.x - A.x, A.x - P.x), vec3(B.y - A.y, C.y - A.y, A.y - P.y));

	// I don't understand this part, why the triangle is degenerate?
	if (fabs(u.z) < 1.f)
		return vec3(-1.f);

	return { 1.f - (u.x + u.y) / u.z, u.x / u.z, u.y / u.z };
}

void drawTriangle(Framebuffer& fb, Shader& shader, bool depthTest, vec3 v1f, vec3 v2f, vec3 v3f)
{
	ivec2 v1 = ivec2(v1f.x, v1f.y) + 0.5f;
	ivec2 v2 = ivec2(v2f.x, v2f.y) + 0.5f;
	ivec2 v3 = ivec2(v3f.x, v3f.y) + 0.5f;

	// finding the boundig box
	ivec2 bboxStart;
	ivec2 bboxEnd;
	bboxStart.x = min(min(v1.x, v2.x), v3.x);
	bboxStart.y = min(min(v1.y, v2.y), v3.y);
	bboxEnd.x = max(max(v1.x, v2.x), v3.x);
	bboxEnd.y = max(max(v1.y, v2.y), v3.y);

	// clipping, todo: (we shouldn't do this here)
	bboxStart.x = max(0, bboxStart.x);
	bboxStart.y = max(0, bboxStart.y);
	bboxEnd.x = min(fb.size.x - 1, bboxEnd.x);
	bboxEnd.y = min(fb.size.y - 1, bboxEnd.y);

	for (int x = bboxStart.x; x <= bboxEnd.x; ++x)
	{
		for (int y = bboxStart.y; y <= bboxEnd.y; ++y)
		{
			const ivec2 p = { x, y };
			const vec3 b = getBarycentric(v1, v2, v3, p);

			// check if the barycentric coordinates are valid
			if (b.x < 0.f || b.y < 0.f || b.z < 0.f)
				continue;

			const float depth = b.x * v1f.z + b.y * v2f.z + b.z * v3f.z;

			if (setPxDepth(fb, p, depth) || !depthTest)
				setPx(fb, p, shader.fragment(b));
		}
	}
}

// using 'shaders' instead of inline code visibly slows down the execution
// todo: linear interpolation with perspective division
void draw(const RenderCommand& rndCmd)
{
	for (int faceIdx = 0; faceIdx < rndCmd.numFraces; ++faceIdx)
	{
		// h for homogeneous
		vec4 hPositions[3];

		for (int i = 0; i < 3; ++i)
		{
			hPositions[i] = rndCmd.shader->vertex(faceIdx, i);
		}

		// todo:
		// here we are in the clip space, we should clip against the w component
		// (remember about z)

		// clip space (perspective divide) -> NDC
		vec3 positions[3];
		for (int i = 0; i < 3; ++i)
		{
			positions[i] = vec3(hPositions[i] / hPositions[i].w);
		}

		// NDC (viewport transform) -> window space (face culling happens here)
		for (int i = 0; i < 3; ++i)
		{
			vec3& v = positions[i];

			v.y *= -1.f; // this is different than in OpenGL,
			             // but I like to have 0,0 coordinate in the top-left corner (like in DirectX)

			v = (v + 1.f) / 2.f;

			v.x *= (rndCmd.fb->size.x - 1.f);
			v.y *= (rndCmd.fb->size.y - 1.f);
		}

		if(!rndCmd.wireframe)
		{
			// face culling, counter-clockwise winding order

			// negate because we inverted the y in the viewport transformation
			// I hope this makes sense
			const bool frontFace = dot(-cross(positions[1] - positions[0], positions[2] - positions[0]),
				vec3(0.f, 0.f, 1.f)) > 0.f;

			bool render = false;

			if (frontFace && !rndCmd.cullFrontFaces)
				render = true;
			else if (!frontFace && !rndCmd.cullBackFaces)
				render = true;

			if (!render)
				continue;

        // rasterization
			drawTriangle(*rndCmd.fb, *rndCmd.shader, rndCmd.depthTest, positions[0], positions[1], positions[2]);
		}
		else
		{
			for (int i = 0; i < 3; ++i)
				drawLine(*rndCmd.fb, positions[i], positions[(i + 1) % 3], { 1.f, 0.f, 1.f });
		}
	}
}

// BUG: z-fighting (at least visually) at certain angles
vec4 Shader1::vertex(int faceIdx, int vIdx)
{
	vec3 pos = model->positions[model->faces[faceIdx].indices[vIdx].position];
	pos.x = pos.x * cos + pos.z * sin;
	pos.z = pos.x * -sin + pos.z * cos;

	vNormals[vIdx] = model->normals[model->faces[faceIdx].indices[vIdx].normal];
	vec3& n = vNormals[vIdx];
	n.x = n.x * cos + n.z * sin;
	n.z = n.x * -sin + n.z * cos;

	// we don't use the projection matrix yet, which flips the z so we have to do this here
	// (to keep depth test correct)
	return vec4(pos.x, pos.y, -pos.z, 1.f);
}

vec3 Shader1::fragment(vec3 b)
{
	vec3 n = normalize(vNormals[0] * b.x + vNormals[1] * b.y + vNormals[2] * b.z);
	vec3 lightDir = normalize(vec3(0.f, -1.f, -0.5f));
	float intensity = max(0.f, dot(n, -lightDir));
	vec3 color(1.f);

	if (style2)
	{
		if (intensity > 0.85f) intensity = 1.f;
		else if (intensity > 0.6f) intensity = 0.8f;
		else if (intensity > 0.45f) intensity = 0.6f;
		else if (intensity > 0.3f) intensity = 0.45f;
		else if (intensity > 0.15f) intensity = 0.3f;
		else intensity = 0.f;

		color = { 1.f, 0.6f, 0.f };
	}

	return color * intensity;
}


class ShaderTest : public Shader
{
public:
	vec4 vertex(int faceIdx, int vIdx) override
	{
		vec3 p = positions[vIdx];
		return vec4(p.x, p.y, -p.z, 1.f);
	}

	vec3 fragment(vec3 b) override
	{
		return b.x * colors[0] + b.y * colors[1] + b.z * colors[2];
	}

	// CCW winding
	vec3 positions[3] = { {-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {0.f, 1.f, 0.f} };
	vec3 colors[3] = { {1.f, 0.f, 0.f }, { 0.f, 1.f, 0.f }, { 0.f, 0.f, 1.f } };
};

Rasterizer::Rasterizer()
{
	glBuffers_ = createGLBuffers();
	glGenTextures(1, &glTexture_);
	glBindTexture(GL_TEXTURE_2D, glTexture_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

	loadModel(model_, "res/diablo3_pose/diablo3_pose.obj");

	shader_.model = &model_;

	rndCmd_.fb = &framebuffer_;
	rndCmd_.shader = &shader_;
	rndCmd_.numFraces = model_.faces.size();
}

Rasterizer::~Rasterizer()
{
	deleteGLBuffers(glBuffers_);
	glDeleteTextures(1, &glTexture_);
}

void Rasterizer::processInput(const Array<WinEvent>& events)
{
	for (const WinEvent& e : events)
	{
		if (e.type == WinEvent::Type::Key && e.key.key == GLFW_KEY_ESCAPE && e.key.action == GLFW_PRESS)
			frame_.popMe = true;
	}
}

void Rasterizer::render(GLuint program)
{
	framebuffer_.colorArray.resize(frame_.fbSize.x * frame_.fbSize.y);
	framebuffer_.depthArray.resize(frame_.fbSize.x * frame_.fbSize.y);
	framebuffer_.size = ivec2(frame_.fbSize);

	static vec3 clearColor(0.f);

	for (RGB8& px : framebuffer_.colorArray)
		px = toRGB8(clearColor);

	for (float& v : framebuffer_.depthArray)
		v = FLT_MAX;

	{
		static float time = 0.f;
		time += frame_.time / 6.f;
		shader_.sin = sinf(time);
		shader_.cos = cosf(time);
	}

	// here the magic happens
	draw(rndCmd_);

	bindProgram(program);
	uniform1i(program, "mode", FragmentMode::Texture);
	uniform2f(program, "cameraPos", 0.f, 0.f);
	uniform2f(program, "cameraSize", frame_.fbSize);

	glBindTexture(GL_TEXTURE_2D, glTexture_);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, frame_.fbSize.x, frame_.fbSize.y, 0, GL_RGB, GL_UNSIGNED_BYTE,
		framebuffer_.colorArray.data());

	Rect rect;
	rect.pos = { 0.f, 0.f };
	rect.size = frame_.fbSize;

	updateGLBuffers(glBuffers_, &rect, 1);
	renderGLBuffers(glBuffers_, 1);

	ImGui::Begin("info");
	ImGui::Text("press Esc to exit");
	ImGui::ColorEdit3("clear color", &clearColor.x);
	ImGui::Text("triangles: %d", model_.faces.size());
	ImGui::Checkbox("wireframe", &rndCmd_.wireframe);

	if (ImGui::TreeNode("cull (counter-clockwise winding order"))
	{
		ImGui::Checkbox("back-faces", &rndCmd_.cullBackFaces);
		ImGui::Checkbox("front-faces", &rndCmd_.cullFrontFaces);
		ImGui::TreePop();
	}

	static bool test = false;
	static ShaderTest tshader;

	if (ImGui::TreeNode("test"))
	{
		ImGui::Checkbox("enable", &test);
		ImGui::Text("counter-clockwise winding");
		ImGui::Text("v0 (left)   - red");
		ImGui::Text("v1 (right)  - green");
		ImGui::Text("v2 (middle) - blue");
		ImGui::TreePop();
	}

	if (test)
	{
		rndCmd_.shader = &tshader;
		rndCmd_.numFraces = 1;
	}
	else
	{
		rndCmd_.shader = &shader_;
		rndCmd_.numFraces = model_.faces.size();
	}

	ImGui::Checkbox("depth test", &rndCmd_.depthTest);
	ImGui::Checkbox("style2", &shader_.style2);

	ImGui::End();
}
