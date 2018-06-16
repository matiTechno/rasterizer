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

	// calculate tangents
	for (Face& face : model.faces)
	{
		vec3 dPos1;
		vec3 dPos2;
		{
			const vec3 v0 = model.positions[face.indices[0].position];
			const vec3 v1 = model.positions[face.indices[1].position];
			const vec3 v2 = model.positions[face.indices[2].position];
			
			dPos1 = v1 - v0;
			dPos1 = v2 - v0;
		}

		vec2 dUV1;
		vec2 dUV2;
		{
			const vec2 uv0 = model.texCoords[face.indices[0].texCoord];
			const vec2 uv1 = model.texCoords[face.indices[1].texCoord];
			const vec2 uv2 = model.texCoords[face.indices[2].texCoord];

			dUV1 = uv1 - uv0;
			dUV2 = uv2 - uv0;
		}

		// this is solving the following linear equation:
		// dPos1 = dUV1.x * T + dUV1.y * B 
		// dPos2 = dUV2.x * T + dUV2.y * B 

		const float r = 1.f / (dUV1.x * dUV2.y - dUV1.y * dUV2.x);
		const vec3 tangent = (dPos1 * dUV2.y - dPos2 * dUV1.y) * r;

		model.tangents.pushBack(tangent);

		for (int i = 0; i < 3; ++i)
			face.indices[i].tangent = model.tangents.size() - 1;
	}
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

	color.x = max(min(color.x, 1.f), 0.f);
	color.y = max(min(color.y, 1.f), 0.f);
	color.z = max(min(color.z, 1.f), 0.f);

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

	// clamping, todo: the same as in drawTriangle
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

	// clamping, todo: we shouldn't do this here, it's a job for a clipper, also clamping deforms
	// triangles which is bad
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
		// this is very 'incomplete' implementation...
		bool clip = false;
		for (int i = 0; i < 3; ++i)
		{
			bool coordOutOfRange = true;
			for (vec4& pos : hPositions)
			{
				// if a ith-coordinate of a pos is in the visible range then ...
				if (-pos.w <= pos[i] && pos[i] <= pos.w)
				{
					coordOutOfRange = false;
					break;
				}
			}

			if (coordOutOfRange)
			{
				clip = true;
				break;
			}
		}

		if (clip)
			continue;

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

			if (rndCmd.shader->tangentDebug.enable)
			{
				for (int vIdx = 0; vIdx < 3; ++vIdx)
				{
					// to NDC
					vec3 TBNPositions[3];
					for (int i = 0; i < 3; ++i)
					{
						vec4 hPos = rndCmd.shader->tangentDebug.positions[vIdx][i];
						TBNPositions[i] = vec3(hPos / hPos.w);
					}
					
					// to window
					for (int i = 0; i < 3; ++i)
					{
						vec3& v = TBNPositions[i];
						v.y *= -1.f;
						v = (v + 1.f) / 2.f;
						v.x *= (rndCmd.fb->size.x - 1.f);
						v.y *= (rndCmd.fb->size.y - 1.f);
					}

					drawLine(*rndCmd.fb, positions[vIdx], TBNPositions[0], { 1.f, 0.f, 0.f }); // tangent
					drawLine(*rndCmd.fb, positions[vIdx], TBNPositions[1], { 0.f, 1.f, 0.f }); // bitangent
					drawLine(*rndCmd.fb, positions[vIdx], TBNPositions[2], { 0.f, 0.f, 1.f }); // normal
				}
			}
		}
		else
		{
			for (int i = 0; i < 3; ++i)
				drawLine(*rndCmd.fb, positions[i], positions[(i + 1) % 3], { 1.f, 0.f, 1.f });
		}
	}
}

vec3 interpolate(const vec3* verts, vec3 baricentricCoords)
{
	vec3 v(0.f);
	for (int i = 0; i < 3; ++i)
	{
		v += verts[i] * baricentricCoords[i];
	}
	return v;
}

vec2 interpolate(const vec2* verts, vec3 baricentricCoords)
{
	vec2 v(0.f);
	for (int i = 0; i < 3; ++i)
	{
		v += verts[i] * baricentricCoords[i];
	}
	return v;
}

mat3 interpolate(const mat3* mats, vec3 baricentricCoords)
{
	// todo: better matrix initialization, something like in glm
	mat3 m;
	m.i = vec3(0.f);
	m.j = vec3(0.f);
	m.k = vec3(0.f);

	for (int i = 0; i < 3; ++i)
		for (int x = 0; x < 3; ++x)
			for (int y = 0; y < 3; ++y)
				m[x][y] += mats[i][x][y] * baricentricCoords[i];

	return m;
}

// todo: normals need to go under a special tranformation: transpose(inverse(modelM)) * n,
// so the non-uniform scaling operation will preserve normal direction
vec4 Shader1::vertex(int faceIdx, int vIdx)
{
	{
		// todo: re-orthogonalize tangent with respect to normal
		vec3 tangent = model->tangents[model->faces[faceIdx].indices[vIdx].tangent];
		tangent = normalize(vec3(mat.model * vec4(tangent, 0.f))); // 0.f to remove the translation part

		vec3 normal = model->normals[model->faces[faceIdx].indices[vIdx].normal];
		normal = normalize(vec3(mat.model * vec4(normal, 0.f)));

		const vec3 bitangent = cross(normal, tangent);

		v.TBN[vIdx].i = tangent;
		v.TBN[vIdx].j = bitangent;
		v.TBN[vIdx].k = normal;

	}

	v.texCoords[vIdx] = model->texCoords[model->faces[faceIdx].indices[vIdx].texCoord];

	{
		vec3 pos = model->positions[model->faces[faceIdx].indices[vIdx].position];
		vec4 worldPos = mat.model * vec4(pos, 1.f);

		if (tangentDebug.enable)
		{
			for(int i = 0; i < 3; ++i)
				tangentDebug.positions[vIdx][i] = mat.projection * mat.view * (vec4(v.TBN[vIdx][i] / 20.f, 0.f) + worldPos);
		}

		v.positions[vIdx] = vec3(worldPos);
		return mat.projection * mat.view * worldPos;
	}
}

vec3 Shader1::fragment(vec3 b)
{
	const vec2 tc = interpolate(v.texCoords, b);

	mat3 TBN = interpolate(v.TBN, b);
	for (int i = 0; i < 3; ++i) TBN[i] = normalize(TBN[i]);

	vec3 n;

	if (useNormalMap)
	{
		n = model->tangentNormalMap.sample(tc) * 2.f - 1.f; 
		n = normalize(TBN * n);
	}
	else
		n = TBN.k;

	light.dir = normalize(light.dir);
	float intensity = max(0.f, dot(n, -light.dir));

	if (style2)
	{
		if (intensity > 0.85f) intensity = 1.f;
		else if (intensity > 0.6f) intensity = 0.8f;
		else if (intensity > 0.45f) intensity = 0.6f;
		else if (intensity > 0.3f) intensity = 0.45f;
		else if (intensity > 0.15f) intensity = 0.3f;
		else intensity = 0.f;
		return vec3(1.f, 0.6f, 0.f) * intensity;
	}
	
	// Phong shading

	vec3 diffuseSample = model->diffuseTexture.sample(tc);
	vec3 diffuse = 0.7f * diffuseSample * intensity;
	vec3 ambient = 0.15f * diffuseSample;

	// I'm not sure if I'm using the specular map correctly here
	// in the tinyrender tutorial: specular map = shininess
	// todo: Blinn-Phong
	const vec3 reflectedLight = reflect(light.dir, n);
	const vec3 fragPos = interpolate(v.positions, b);
	const vec3 viewDir = normalize(fragPos - cameraPos);
	const float shininess = 10.f;
	const vec3 specular = model->specularTexture.sample(tc)
		* powf(max(0.f, dot(-viewDir, reflectedLight)), shininess);

	const vec3 glow = model->glowTexture.sample(tc);

	return diffuse + specular + ambient + glow;
}

class ShaderTest : public Shader
{
public:
	vec4 vertex(int faceIdx, int vIdx) override
	{
		vec3 p = positions[vIdx];
		return vec4(p.x, p.y, p.z, 1.f);
	}

	vec3 fragment(vec3 b) override
	{
		return interpolate(colors, b);
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
	loadBitmapFromFile(model_.diffuseTexture, "res/diablo3_pose/diablo3_pose_diffuse.tga");
	loadBitmapFromFile(model_.specularTexture, "res/diablo3_pose/diablo3_pose_spec.tga");
	loadBitmapFromFile(model_.glowTexture, "res/diablo3_pose/diablo3_pose_glow.tga");
	loadBitmapFromFile(model_.tangentNormalMap, "res/diablo3_pose/diablo3_pose_nm_tangent.tga");

	shader_.model = &model_;

	rndCmd_.fb = &framebuffer_;
	rndCmd_.shader = &shader_;
	rndCmd_.numFraces = model_.faces.size();
}

Rasterizer::~Rasterizer()
{
	deleteBitmap(model_.diffuseTexture);
	deleteBitmap(model_.specularTexture);
	deleteBitmap(model_.glowTexture);
	deleteBitmap(model_.tangentNormalMap);
	deleteGLBuffers(glBuffers_);
	glDeleteTextures(1, &glTexture_);
}

void Rasterizer::processInput(const Array<WinEvent>& events)
{
	for (const WinEvent& e : events)
	{
		if (e.type == WinEvent::Type::Key && e.key.key == GLFW_KEY_ESCAPE && e.key.action == GLFW_PRESS)
			frame_.popMe = true;

		if(useFpsCamera_) camera_.processEvent(e);
		else arcball_.processEvent(e);
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

	static bool rotate = false;
	static float angle = 0.f;

	if (rotate)
		angle += frame_.time * 8.f;

	if (useFpsCamera_) camera_.update(frame_.time);
	else arcball_.update();

	// setting up the shader

	shader_.mat.model = rotateY(angle);
	float aspect = float(framebuffer_.size.x) / framebuffer_.size.y;
	shader_.mat.projection = perspective(45.f, aspect, 0.1f, 50.f);
	shader_.light.dir = { 0.f, -1.f, -1.f };

	shader_.mat.view = useFpsCamera_ ? camera_.view : arcball_.view;
	shader_.cameraPos = useFpsCamera_ ? camera_.pos: arcball_.pos;

	// todo: draw the light source
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
	ImGui::Checkbox("use fps camera", &useFpsCamera_);

	if(useFpsCamera_) camera_.imgui();
	else arcball_.imgui();

	ImGui::ColorEdit3("clear color", &clearColor.x);
	ImGui::Text("model faces: %d", model_.faces.size());
	ImGui::Checkbox("rotate model", &rotate);
	ImGui::Checkbox("wireframe", &rndCmd_.wireframe);

	ImGui::Text("cull (counter-clockwise winding order)");
	ImGui::Checkbox("back-faces", &rndCmd_.cullBackFaces);
	ImGui::Checkbox("front-faces", &rndCmd_.cullFrontFaces);

	ImGui::Checkbox("depth test", &rndCmd_.depthTest);
	ImGui::Checkbox("style2", &shader_.style2);
	ImGui::Checkbox("use normal map", &shader_.useNormalMap);
	ImGui::Checkbox("debug tangents", &shader_.tangentDebug.enable);

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

	ImGui::End();
}

Camera3d::Camera3d()
{
	controls_[Forward] = GLFW_KEY_W;
	controls_[Back] = GLFW_KEY_S;
	controls_[Left] = GLFW_KEY_A;
	controls_[Right] = GLFW_KEY_D;
	controls_[Up] = GLFW_KEY_SPACE;
	controls_[Down] = GLFW_KEY_LEFT_SHIFT;
	controls_[ToggleMouseCapture] = GLFW_KEY_1;
}

void Camera3d::captureMouse()
{
	glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	firstCursorEvent_ = true;
	mouseCapture_ = true;
}

void Camera3d::processEvent(const WinEvent& event)
{
	if (event.type == WinEvent::Key)
	{
		int idx = -1;
		for (int i = 0; i < NumControls; ++i)
		{
			if (event.key.key == controls_[i])
			{
				idx = i;
				break;
			}
		}

		if (idx == -1)
			return;

		if (event.key.action == GLFW_PRESS)
		{
			keys_.pressed[idx] = true;
			keys_.held[idx] = true;

			if (idx == ToggleMouseCapture)
			{
				if (!mouseCapture_)
					captureMouse();
				else
				{
					mouseCapture_ = false;
					glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				}
			}
		}
		else if (event.key.action == GLFW_RELEASE)
		{
			keys_.held[idx] = false;
		}
	}
	else if (event.type == WinEvent::Cursor && mouseCapture_)
	{
		const vec2 offset = event.cursor.pos - cursorPos_;
		cursorPos_ = event.cursor.pos;

		if (!firstCursorEvent_)
		{
			pitch -= offset.y * sensitivity;
			pitch = min(89.f, pitch);
			pitch = max(-89.f, pitch);
			yaw = fmodf(yaw - offset.x * sensitivity, 360.f);
		}
		else
			firstCursorEvent_ = false;
	}
}

void Camera3d::update(const float time)
{
	up = normalize(up);

	const vec3 dir = normalize( vec3(
		cosf(toRadians(pitch)) * sinf(toRadians(yaw)) * -1.f,
		sinf(toRadians(pitch)),
		cosf(toRadians(pitch)) * cosf(toRadians(yaw)) * -1.f));

	vec3 moveDir(0.f);

	{
		const auto forwardDir = forwardXZonly ? normalize(vec3(dir.x, 0.f, dir.z)) : dir;
		if (cActive(Forward)) moveDir += forwardDir;
		if (cActive(Back)) moveDir -= forwardDir;
	}
	{
		const auto right = normalize(cross(dir, up));
		if (cActive(Left)) moveDir -= right;
		if (cActive(Right)) moveDir += right;
	}

	if (cActive(Up)) moveDir += up;
	if (cActive(Down)) moveDir -= up;

	if (length(moveDir) != 0.f)
		normalize(moveDir);

	pos += moveDir * speed * time;
	view = lookAt(pos, pos + dir, up);

	for (auto& key : keys_.pressed)
		key = false;
}

void Camera3d::imgui()
{
	ImGui::Text("enable / disable mouse capture - 1");
	ImGui::Text("pitch / yaw - mouse");
	ImGui::Text("move - wsad, space (up), lshift (down)");
	ImGui::Text("pos: x: %.3f, y: %.3f, z: %.3f", pos.x, pos.y, pos.z);
	ImGui::Text("pitch: %.3f, yaw: %.3f", pitch, yaw);
	ImGui::Checkbox("disable flying with WS", &forwardXZonly);
}

void ArcballCamera::processEvent(const WinEvent& event)
{
	if (event.type == WinEvent::Cursor)
	{
		if(buttonPressed_)
			cursorPosDelta_ += event.cursor.pos - cursorPos_;

		cursorPos_ = event.cursor.pos;
	}
	else if (event.type == WinEvent::MouseButton && event.mouseButton.button == GLFW_MOUSE_BUTTON_LEFT)
	{
		buttonPressed_ = event.mouseButton.action == GLFW_PRESS;
	}
	else if (event.type == WinEvent::Scroll)
	{
		scrollDelta_ += event.scroll.offset.y;
	}
}

void ArcballCamera::update()
{
	{
		const vec3 move = normalize(-pos) * 0.2f * scrollDelta_ * zoomSensitivity;
		const vec3 prevPos = pos;
		pos += move;

		if (dot(prevPos, pos) < 0.f)
			pos = prevPos; // not vec3(0.f) because then we lose the direction
	}

	// todo:
	// implement view matrix with rotations instead of LookAt() like in Sascha's examples
	// (I think it might be much better, camera.pos will be vec3(0.f, 0.f, z) then - quite a different model)

	// I don't know how to do it correctly for both axis with this technique...
	const float angleX = cursorPosDelta_.x * rotateSensitivity * -1.f;
	pos = vec3(rotateY(angleX) * vec4(pos, 1.f));
	view = lookAt(pos, vec3(0.f), { 0.f, 1.f, 0.f });

	cursorPosDelta_ = vec2(0.f);
	scrollDelta_ = 0.f;
}

void ArcballCamera::imgui()
{
	ImGui::Text("Arcball camera - press lmb and move the cursor\n"
		"to rotate around the scene, scroll to zoom");
	ImGui::Text("pos: x: %.3f, y: %.3f, z: %.3f", pos.x, pos.y, pos.z);
}
