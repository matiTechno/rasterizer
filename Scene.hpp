#pragma once

#include "Array.hpp"
#include "float.h"

template<typename T>
inline T max(T a, T b) {return a > b ? a : b;}

template<typename T>
inline T min(T a, T b) {return a < b ? a : b;}

template<typename T>
inline void swapcpy(T&l, T& r)
{
	T temp = l;
	l = r;
	r = temp;
}

using GLuint = unsigned int;

// use on plain C arrays
template<typename T, int N>
constexpr int getSize(T(&)[N])
{
    return N;
}

template<typename T>
struct tvec2
{
    tvec2() = default;
    explicit tvec2(T v): x(v), y(v) {}
    tvec2(T x, T y): x(x), y(y) {}

    template<typename U>
    explicit tvec2(tvec2<U> v): x(v.x), y(v.y) {}

    //                @ const tvec2& ?
    tvec2& operator+=(tvec2 v) {x += v.x; y += v.y; return *this;}
    tvec2& operator+=(T v) {x += v; y += v; return *this;}
    tvec2& operator-=(tvec2 v) {x -= v.x; y -= v.y; return *this;}
    tvec2& operator-=(T v) {x -= v; y -= v; return *this;}
    tvec2& operator*=(tvec2 v) {x *= v.x; y *= v.y; return *this;}
    tvec2& operator*=(T v) {x *= v; y *= v; return *this;}
    tvec2& operator/=(tvec2 v) {x /= v.x; y /= v.y; return *this;}
    tvec2& operator/=(T v) {x /= v; y /= v; return *this;}

    tvec2 operator+(tvec2 v) const {return {x + v.x, y + v.y};}
    tvec2 operator+(T v)     const {return {x + v, y + v};}
    tvec2 operator-(tvec2 v) const {return {x - v.x, y - v.y};}
    tvec2 operator-(T v)     const {return {x - v, y - v};}
	tvec2 operator-()        const {return {-x, -y };}
    tvec2 operator*(tvec2 v) const {return {x * v.x, y * v.y};}
    tvec2 operator*(T v)     const {return {x * v, y * v};}
    tvec2 operator/(tvec2 v) const {return {x / v.x, y / v.y};}
    tvec2 operator/(T v)     const {return {x / v, y / v};}

    bool operator==(tvec2 v) const {return x == v.x && y == v.y;}
    bool operator!=(tvec2 v) const {return !(*this == v);}

    T x;
    T y;
};

template<typename T>
inline tvec2<T> operator*(T scalar, tvec2<T> v) {return v * scalar;}

using ivec2 = tvec2<int>;
using vec2  = tvec2<float>;

template<typename T>
struct tvec3
{
    tvec3() = default;
    explicit tvec3(T v): x(v), y(v), z(v) {}
    tvec3(T x, T y, T z): x(x), y(y), z(z) {}

    template<typename U>
    explicit tvec3(tvec3<U> v): x(v.x), y(v.y), z(v.z) {}

    //                @ const tvec2& ?
	tvec3& operator+=(tvec3 v) { x += v.x; y += v.y; z += v.z; return *this; }
	tvec3& operator+=(T v) { x += v; y += v; z += v; return *this; }
	tvec3& operator-=(tvec3 v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	tvec3& operator-=(T v) { x -= v; y -= v; z -= v; return *this; }
	tvec3& operator*=(tvec3 v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	tvec3& operator*=(T v) { x *= v; y *= v; z *= v; return *this; }
	tvec3& operator/=(tvec3 v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
	tvec3& operator/=(T v) { x /= v; y /= v; z /= v; return *this; }

    tvec3 operator+(tvec3 v) const {return {x + v.x, y + v.y, z + v.z};}
    tvec3 operator+(T v)     const {return {x + v, y + v, z + v};}
    tvec3 operator-(tvec3 v) const {return {x - v.x, y - v.y, z - v.z};}
    tvec3 operator-(T v)     const {return {x - v, y - v, z - v};}
	tvec3 operator-()        const {return {-x, -y, -z};}
    tvec3 operator*(tvec3 v) const {return {x * v.x, y * v.y, z * v.z};}
    tvec3 operator*(T v)     const {return {x * v, y * v, z * v};}
    tvec3 operator/(tvec3 v) const {return {x / v.x, y / v.y, z / v.z};}
    tvec3 operator/(T v)     const {return {x / v, y / v, z / v};}

    bool operator==(tvec3 v) const {return x == v.x && y == v.y && z == v.z;}
    bool operator!=(tvec3 v) const {return !(*this == v);}

    T x;
    T y;
	T z;
};

template<typename T>
inline tvec3<T> operator*(T scalar, tvec3<T> v) {return v * scalar;}

using ivec3 = tvec3<int>;
using vec3  = tvec3<float>;

struct vec4
{
    float x;
    float y;
    float z;
    float w;
};

struct FragmentMode
{
    enum
    {
        Color = 0,
        Texture = 1
    };
};

struct Texture
{
    ivec2 size;
    GLuint id;
};

// @TODO(matiTechno)
// add origin for rotation (needed to properly rotate a text)
struct Rect
{
    vec2 pos;
    vec2 size;
    vec4 color = {1.f, 1.f, 1.f, 1.f};
    vec4 texRect = {0.f, 0.f, 1.f, 1.f};
    float rotation = 0.f;
};

// @ better naming?
struct GLBuffers
{
    GLuint vao;
    GLuint vbo;
    GLuint rectBo;
};

struct WinEvent
{
    enum Type
    {
        Nil,
        Key,
        Cursor,
        MouseButton,
        Scroll
    };

    Type type;

    // glfw values
    union
    {
        struct
        {
            int key;
            int action;
            int mods;
        } key;

        struct
        {
            vec2 pos;
        } cursor;

        struct
        {
            int button;
            int action;
            int mods;
        } mouseButton;

        struct
        {
            vec2 offset;
        } scroll;
    };
};

// call bindeProgram() first
void uniform1i(GLuint program, const char* name, int i);
void uniform1f(GLuint program, const char* name, float f);
void uniform2f(GLuint program, const char* name, float f1, float f2);
void uniform2f(GLuint program, const char* name, vec2 v);
void uniform3f(GLuint program, const char* name, float f1, float f2, float f3);
void uniform4f(GLuint program, const char* name, float f1, float f2, float f3, float f4);
void uniform4f(GLuint program, const char* name, vec4 v);

void bindTexture(const Texture& texture, GLuint unit = 0);
// delete with deleteTexture()
Texture createTextureFromFile(const char* filename);
void deleteTexture(const Texture& texture);

// returns 0 on failure
// program must be deleted with deleteProgram() (if != 0)
GLuint createProgram(const char* vertexSrc, const char* fragmentSrc);
void deleteProgram(GLuint program);
void bindProgram(const GLuint program);

// delete with deleteGLBuffers()
GLBuffers createGLBuffers();
// @TODO(matiTechno): we might want updateSubGLBuffers()
void updateGLBuffers(GLBuffers& glBuffers, const Rect* rects, int count);
// call bindProgram() first
void renderGLBuffers(GLBuffers& glBuffers, int numRects);
void deleteGLBuffers(GLBuffers& glBuffers);

struct Camera
{
    vec2 pos;
    vec2 size;
};

Camera expandToMatchAspectRatio(Camera camera, vec2 viewportSize);

// [min, max]
float getRandomFloat(float min, float max);
int getRandomInt(int min, int max);

class Scene
{
public:
    virtual ~Scene() = default;
    virtual void processInput(const Array<WinEvent>& events) {(void)events;}
    virtual void update() {}
    virtual void render(GLuint program) {(void)program;}

    struct
    {
        // these are set in the main loop before processInput() call
        float time;  // seconds
        vec2 fbSize; // fb = framebuffer

        // @TODO(matiTechno): bool updateWhenNotTop = false;
        bool popMe = false;
        Scene* newScene = nullptr; // assigned ptr must be returned by new
                                   // game loop will call delete on it
    } frame_;
};

struct RGB8
{
	unsigned char r, g, b;
};

struct Index
{
	int position, texCoord, normal;
};

struct Face
{
	Index indices[3];
};

struct Model
{
	Array<Face> faces;
	Array<vec3> positions;
	Array<vec2> texCoords;
	Array<vec3> normals;
};

struct Framebuffer
{
	Array<RGB8> colorArray;
	Array<float> depthArray;
	ivec2 size;
};

class Rasterizer: public Scene
{
public:
    Rasterizer();
    ~Rasterizer() override;
    void processInput(const Array<WinEvent>& events) override;
    void render(GLuint program) override;

private:
	Framebuffer framebuffer_;
	GLuint glTexture_;
	GLBuffers glBuffers_;
	Model model_;
};
