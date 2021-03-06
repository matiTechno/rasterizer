#define _CRT_SECURE_NO_WARNINGS
#include "glad.h"
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "imgui/imgui.h"
//#include "imgui/imgui_impl_glfw_gl3.h"
#include "Array.hpp"
#include <string.h>
#include <assert.h>
#include "Scene.hpp"

// unity build
#include "Rasterizer.cpp"
#include "glad.c"
#include "imgui/imgui.cpp"
#include "imgui/imgui_demo.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_impl_glfw_gl3.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void loadBitmapFromFile(Bitmap& bitmap, const char* filename)
{
	stbi_set_flip_vertically_on_load(true);
	bitmap.data = (RGB8*)stbi_load(filename, &bitmap.size.x, &bitmap.size.y, nullptr, 3);

	if (!bitmap.data)
		printf("stbi_load() failed: %s\n", filename);
}

void deleteBitmap(Bitmap& bitmap)
{
	if(bitmap.data)
		stbi_image_free(bitmap.data);
}


static GLint getUniformLocation(GLuint program, const char* const name)
{
    GLint loc = glGetUniformLocation(program, name);
    if(loc == -1)
        printf("program = %u: unfiform '%s' is inactive\n", program, name);
    return loc;
}

// @TODO(matiTechno): inline?

void uniform1i(const GLuint program, const char* const name, const int i)
{
    glUniform1i(getUniformLocation(program, name), i);
}

void uniform1f(const GLuint program, const char* const name, const float f)
{
    glUniform1f(getUniformLocation(program, name), f);
}

void uniform2f(const GLuint program, const char* const name, const float f1, const float f2)
{
    glUniform2f(getUniformLocation(program, name), f1, f2);
}

void uniform2f(const GLuint program, const char* const name, const vec2 v)
{
    glUniform2f(getUniformLocation(program, name), v.x, v.y);
}

// @TODO(matiTechno): vec3 Type + uniform3f(vec3)?

void uniform3f(const GLuint program, const char* const name, const float f1, const float f2,
               const float f3)
{
    glUniform3f(getUniformLocation(program, name), f1, f2, f3);
}

void uniform4f(const GLuint program, const char* const name, const float f1, const float f2,
               const float f3, const float f4)
{
    glUniform4f(getUniformLocation(program, name), f1, f2, f3, f4);
}

void uniform4f(const GLuint program, const char* const name, const vec4 v)
{
    glUniform4f(getUniformLocation(program, name), v.x, v.y, v.z, v.w);
}

static void errorCallback(const int error, const char* const description)
{
    (void)error;
    printf("GLFW error: %s\n", description);
}

void bindTexture(const Texture& texture, const GLuint unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture.id);
}

// @TODO(matiTechno): functions for setting texture sampling type
// delete with deleteTexture()
static Texture createDefaultTexture()
{
    Texture tex;
    glGenTextures(1, &tex.id);
    bindTexture(tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex.size = {1, 1};
    const unsigned char color[] = {0, 255, 0, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex.size.x, tex.size.y, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, color);
    
    return tex;
}

// delete with deleteTexture()
Texture createTextureFromFile(const char* const filename)
{
    Texture tex;
    glGenTextures(1, &tex.id);
    bindTexture(tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    unsigned char* const data = stbi_load(filename, &tex.size.x, &tex.size.y,
                                          nullptr, 4);

    if(!data)
    {
        printf("stbi_load() failed: %s\n", filename);
        tex.size = {1, 1};
        const unsigned char color[] = {0, 255, 0, 255};
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex.size.x, tex.size.y, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, color);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex.size.x, tex.size.y, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
    }

    return tex;
}

void deleteTexture(const Texture& texture)
{
    glDeleteTextures(1, &texture.id);
}

const char* const vertexSrc = R"(
#version 330

layout(location = 0) in vec4 aVertex;

// instanced
layout(location = 1) in vec2 aiPos;
layout(location = 2) in vec2 aiSize;
layout(location = 3) in vec4 aiColor;
layout(location = 4) in vec4 aiTexRect;
layout(location = 5) in float aiRotation;

uniform vec2 cameraPos;
uniform vec2 cameraSize;

out vec4 vColor;
out vec2 vTexCoord;

void main()
{
    vColor = aiColor;
    vTexCoord = aVertex.zw * aiTexRect.zw + aiTexRect.xy;

    vec2 aPos = aVertex.xy;
    vec2 pos;

    // rotation
    float s = sin(aiRotation);
    float c = cos(aiRotation);
    pos.x = aPos.x * c - aPos.y * s;
    // -(...) because in our coordinate system y grows down
    pos.y = -(aPos.x * s + aPos.y * c);

    // convert to world coordinates
    //           see static vbo buffer
    pos = (pos + vec2(0.5)) * aiSize + aiPos;

    // convert to clip space
    pos = (pos - cameraPos) * vec2(2.0) / cameraSize
          + vec2(-1.0);
    // in OpenGL y grows up, we have to do the flipping
    pos.y *= -1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

const char* const fragmentSrc = R"(
#version 330

in vec4 vColor;
in vec2 vTexCoord;

uniform sampler2D sampler;
uniform int mode = 0;

out vec4 color;

void main()
{
    color = vColor;

    if(mode == 1)
    {
        vec4 texColor = texture(sampler, vTexCoord);
        // premultiply alpha
        // texColor.rgb *= texColor.a;
        color *= texColor;
    }
}
)";

// returns true on error
static bool isCompileError(const GLuint shader)
{
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    
    if(success == GL_TRUE)
        return false;
    else
    {
        char buffer[512];
        glGetShaderInfoLog(shader, sizeof(buffer), nullptr, buffer);
        printf("glCompileShader() error:\n%s\n", buffer);
        return true;
    }
}

// returns 0 on failure
// program must be deleted with deleteProgram() (if != 0)
GLuint createProgram(const char* const vertexSrc, const char* const fragmentSrc)
{
    const GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexSrc, nullptr);
    glCompileShader(vertex);

    const GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentSrc, nullptr);
    glCompileShader(fragment);
    
    {
        const bool vertexError = isCompileError(vertex);
        const bool fragmentError = isCompileError(fragment);

        if(vertexError || fragmentError)
        {
            glDeleteShader(vertex);
            glDeleteShader(fragment);
            return 0;
        }
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDetachShader(program, vertex);
    glDetachShader(program, fragment);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    
    if(success == GL_TRUE)
        return program;
    else
    {
        glDeleteProgram(program);
        char buffer[512];
        glGetProgramInfoLog(program, sizeof(buffer), nullptr, buffer);
        printf("glLinkProgram() error:\n%s\n", buffer);
        return 0;
    }
}

void deleteProgram(const GLuint program)
{
    glDeleteProgram(program);
}

// delete with deleteGLBuffers()
GLBuffers createGLBuffers()
{
    GLBuffers glBuffers;
    glGenVertexArrays(1, &glBuffers.vao);
    glGenBuffers(1, &glBuffers.vbo);
    glGenBuffers(1, &glBuffers.rectBo);

    float vertices[] = 
    {
        -0.5f, -0.5f, 0.f, 1.f,
        0.5f, -0.5f, 1.f, 1.f,
        0.5f, 0.5f, 1.f, 0.f,
        0.5f, 0.5f, 1.f, 0.f,
        -0.5f, 0.5f, 0.f, 0.f,
        -0.5f, -0.5f, 0.f, 1.f
    };

    // static buffer
    glBindBuffer(GL_ARRAY_BUFFER, glBuffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);

    glBindVertexArray(glBuffers.vao);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // dynamic instanced buffer
    glBindBuffer(GL_ARRAY_BUFFER, glBuffers.rectBo);

    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);
    glVertexAttribDivisor(1, 1);
    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Rect), nullptr);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Rect),
                          (const void*)offsetof(Rect, size));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Rect),
                          (const void*)offsetof(Rect, color));
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Rect),
                          (const void*)offsetof(Rect, texRect));
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(Rect),
                          (const void*)offsetof(Rect, rotation));

    return glBuffers;
}

void updateGLBuffers(GLBuffers& glBuffers, const Rect* const rects, const int count)
{
    glBindBuffer(GL_ARRAY_BUFFER, glBuffers.rectBo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Rect) * count, rects, GL_DYNAMIC_DRAW);
}

// @TODO(matiTechno): do we need these?
void bindProgram(const GLuint program)
{
    glUseProgram(program);
}

// call bindProgram() first
void renderGLBuffers(GLBuffers& glBuffers, const int numRects)
{
    glBindVertexArray(glBuffers.vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numRects);
}

void deleteGLBuffers(GLBuffers& glBuffers)
{
    glDeleteVertexArrays(1, &glBuffers.vao);
    glDeleteBuffers(1, &glBuffers.vbo);
    glDeleteBuffers(1, &glBuffers.rectBo);
}

static Array<WinEvent>* eventsPtr;

static void keyCallback(GLFWwindow* const window, const int key, const int scancode,
                             const int action, const int mods)
{
    WinEvent e;
    e.type = WinEvent::Key;
    e.key.key = key;
    e.key.action = action;
    e.key.mods = mods;
    eventsPtr->pushBack(e);

    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

static void cursorPosCallback(GLFWwindow*, const double xpos, const double ypos)
{
    WinEvent e;
    e.type = WinEvent::Cursor;
    e.cursor.pos.x = xpos;
    e.cursor.pos.y = ypos;
    eventsPtr->pushBack(e);
}

static void mouseButtonCallback(GLFWwindow* const window, const int button, const int action,
                                const int mods)
{
    WinEvent e;
    e.type = WinEvent::MouseButton;
    e.mouseButton.button = button;
    e.mouseButton.action = action;
    e.mouseButton.mods = mods;
    eventsPtr->pushBack(e);

    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

static void scrollCallback(GLFWwindow* const window, const double xoffset,
                           const double yoffset)
{
    WinEvent e;
    e.type = WinEvent::Scroll;
    e.scroll.offset.x = xoffset;
    e.scroll.offset.y = yoffset;
    eventsPtr->pushBack(e);

    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

static void charCallback(GLFWwindow* const window, const unsigned int codepoint)
{
    ImGui_ImplGlfw_CharCallback(window, codepoint);
}


Camera expandToMatchAspectRatio(Camera camera, const vec2 viewportSize)
{
    const vec2 prevSize = camera.size;
    const float viewportAspect = viewportSize.x / viewportSize.y;
    const float cameraAspect = camera.size.x / camera.size.y;

    if(viewportAspect > cameraAspect)
    {
        camera.size.x = camera.size.y * viewportAspect;
    }
    else if(viewportAspect < cameraAspect)
    {
        camera.size.y = camera.size.x / viewportAspect;
    }

    camera.pos -= (camera.size - prevSize) / 2.f;
    return camera;
}

// @TODO(matiTechno): do a research on rngs, shuffle bag (rand() might not be good enough)
// [min, max]

float getRandomFloat(const float min, const float max)
{
    assert(min <= max);
    return min + (max - min) * ( float(rand()) / float(RAND_MAX) );
}

int getRandomInt(const int min, const int max)
{
    assert(min <= max);
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

extern GLFWwindow* gWindow;

int main()
{
    glfwSetErrorCallback(errorCallback);

    if(!glfwInit())
        return EXIT_FAILURE;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;
    {
        GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        window = glfwCreateWindow(mode->width, mode->height, "rasterizer", monitor, nullptr);
    }

    if(!window)
    {
        glfwTerminate();
        return EXIT_FAILURE;
    }

	gWindow = window;

    // @TODO(matiTechno): do a research on rngs, shuffle bag (rand() might not be good enough)
    srand(time(nullptr));

    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCharCallback(window, charCallback);

    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, false);
	ImGui::GetStyle().ScaleAllSizes(1.4f);

    GLuint program = createProgram(vertexSrc, fragmentSrc);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Array<WinEvent> events;
    events.reserve(50);
    eventsPtr = &events;

    Scene* scenes[10];
    int numScenes = 1;
	scenes[0] = new Rasterizer;

    struct
    {
        float accumulator = 0.f;
        int frameCount = 0;
        float frameTimes[180] = {}; // ms
    } plot;

    double time = glfwGetTime();

    // for now we will handle only the top scene
    while(!glfwWindowShouldClose(window) && numScenes)
    {
        double newTime = glfwGetTime();
        const float dt = newTime - time;
        time = newTime;

        plot.accumulator += dt;
        ++plot.frameCount;

        if(plot.accumulator >= 0.033f)
        {
            memmove(plot.frameTimes, plot.frameTimes + 1, sizeof(plot.frameTimes) -
                                                          sizeof(float));

            plot.frameTimes[getSize(plot.frameTimes) - 1] = plot.accumulator / plot.frameCount
                                                            * 1000.f;
            plot.accumulator = 0.f;
            plot.frameCount = 0;
        }

        events.clear();
        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        const bool imguiWantMouse = ImGui::GetIO().WantCaptureMouse;
        const bool imguiWantKeyboard = ImGui::GetIO().WantCaptureKeyboard;

        for(WinEvent& e: events)
        {
            if(imguiWantMouse && ((e.type == WinEvent::MouseButton &&
                                      e.mouseButton.action != GLFW_RELEASE) ||
                                     e.type == WinEvent::Cursor ||
                                     e.type == WinEvent::Scroll))
                e.type = WinEvent::Nil;

            if(imguiWantKeyboard && (e.type == WinEvent::Key))
                e.type = WinEvent::Nil;
        }

        ivec2 fbSize;
        glfwGetFramebufferSize(window, &fbSize.x, &fbSize.y);
        glViewport(0, 0, fbSize.x, fbSize.y);
        glClear(GL_COLOR_BUFFER_BIT);

        Scene& scene = *scenes[numScenes - 1];
        scene.frame_.time = dt;
        scene.frame_.fbSize.x = fbSize.x;
        scene.frame_.fbSize.y = fbSize.y;
        
        ImGui::Begin("info");
        {
            if(ImGui::Button("quit"))
                glfwSetWindowShouldClose(window, true);

            ImGui::Spacing();
            ImGui::Text("framebuffer size: %d x %d", fbSize.x, fbSize.y);

            float maxTime = 0.f;
            float sum = 0.f;

            for(const float t: plot.frameTimes)
            {
                sum += t;
                maxTime = max(maxTime, t);
            }

            const float avg = sum / getSize(plot.frameTimes);

            ImGui::Spacing();
            ImGui::Text("frame time ms");
            ImGui::PushStyleColor(ImGuiCol_Text, {0.f, 0.85f, 0.f, 1.f});
            ImGui::Text("avg   %.3f (%d)", avg, int(1.f / avg * 1000.f + 0.5f));
            ImGui::PushStyleColor(ImGuiCol_Text, {0.9f, 0.f, 0.f, 1.f});
            ImGui::Text("max   %.3f", maxTime);
            ImGui::PopStyleColor(2);
            ImGui::Spacing();
            ImGui::PlotLines("", plot.frameTimes, getSize(plot.frameTimes), 0, nullptr, 0.f,
                             150.f, {400, 150});

            ImGui::Spacing();
            ImGui::Text("vsync");
            ImGui::SameLine();
            if(ImGui::Button("on "))
                glfwSwapInterval(1);
            
            ImGui::SameLine();

            if(ImGui::Button("off"))
                glfwSwapInterval(0);

        }
        ImGui::End();

        scene.processInput(events);
        scene.update();
        scene.render(program);

        ImGui::Render();
        ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        Scene* newScene = nullptr;
        newScene = scene.frame_.newScene;
        scene.frame_.newScene = nullptr;

        if(scene.frame_.popMe)
        {
            delete &scene;
            --numScenes;
        }

        if(newScene)
        {
            ++numScenes;
            assert(numScenes < getSize(scenes));
            scenes[numScenes - 1] = newScene;
        }
    }

    for(int i = numScenes - 1; i >= 0; --i)
    {
        delete scenes[i];
    }

    deleteProgram(program);
    ImGui_ImplGlfwGL3_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return EXIT_SUCCESS;
}
