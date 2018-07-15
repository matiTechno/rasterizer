#pragma once

#include "Array.hpp"
#include <math.h>

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
struct tvec3;

template<typename T>
struct tvec2;

template<typename T>
struct tvec4
{
	// todo: create from vec3, vec2, ...
	// the same for vec3
    tvec4() = default;
    explicit tvec4(T v): x(v), y(v), z(v), w(v) {}
    tvec4(T x, T y, T z, T w): x(x), y(y), z(z), w(w) {}
	tvec4(const tvec3<T>& v, T w);
	tvec4(T x, const tvec3<T>& v);
	tvec4(const tvec2<T>& v1, const tvec2<T>& v2);
	tvec4(const tvec2<T>& v, T z, T w);
	tvec4(T x, T y, const tvec2<T>& v);

    template<typename U>
    explicit tvec4(tvec4<U> v): x(v.x), y(v.y), z(v.z), w(v.w) {}

    //                @ const tvec4& ?
	tvec4& operator+=(tvec4 v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	tvec4& operator+=(T v) { x += v; y += v; z += v; w += v; return *this; }
	tvec4& operator-=(tvec4 v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
	tvec4& operator-=(T v) { x -= v; y -= v; z -= v; w -= v; return *this; }
	tvec4& operator*=(tvec4 v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
	tvec4& operator*=(T v) { x *= v; y *= v; z *= v; w *= v; return *this; }
	tvec4& operator/=(tvec4 v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w;  return *this; }
	tvec4& operator/=(T v) { x /= v; y /= v; z /= v; w /= v; return *this; }

    tvec4 operator+(tvec4 v) const {return {x + v.x, y + v.y, z + v.z, w + v.w};}
    tvec4 operator+(T v)     const {return {x + v, y + v, z + v, w + v};}
    tvec4 operator-(tvec4 v) const {return {x - v.x, y - v.y, z - v.z, w - v.w};}
    tvec4 operator-(T v)     const {return {x - v, y - v, z - v, w - v};}
	tvec4 operator-()        const {return {-x, -y, -z, -w};}
    tvec4 operator*(tvec4 v) const {return {x * v.x, y * v.y, z * v.z, w * v.w};}
    tvec4 operator*(T v)     const {return {x * v, y * v, z * v, w * v};}
    tvec4 operator/(tvec4 v) const {return {x / v.x, y / v.y, z / v.z, w / v.w};}
    tvec4 operator/(T v)     const {return {x / v, y / v, z / v, w / v};}

	const T&     operator[](int idx)const {return *(&x + idx);}
	      T&     operator[](int idx)      {return *(&x + idx);}

    bool operator==(tvec4 v) const {return x == v.x && y == v.y && z == v.z && w == v.w;}
    bool operator!=(tvec4 v) const {return !(*this == v);}

    T x;
    T y;
	T z;
	T w;
};

template<typename T>
inline tvec4<T> operator*(T scalar, tvec4<T> v) {return v * scalar;}

using ivec4 = tvec4<int>;
using vec4  = tvec4<float>;

template<typename T>
struct tvec3
{
    tvec3() = default;
    explicit tvec3(T v): x(v), y(v), z(v) {}
    tvec3(T x, T y, T z): x(x), y(y), z(z) {}
	tvec3(T x, const tvec2<T>& v);
	tvec3(const tvec2<T>& v, T z);

    template<typename U>
    explicit tvec3(tvec3<U> v): x(v.x), y(v.y), z(v.z) {}

	template<typename U>
	explicit tvec3(tvec4<U> v) : x(v.x), y(v.y), z(v.z) {}

    //                @ const tvec3& ?
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

	const T&     operator[](int idx)const {return *(&x + idx);}
	      T&     operator[](int idx)      {return *(&x + idx);}

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

template<typename T>
struct tvec2
{
    tvec2() = default;
    explicit tvec2(T v): x(v), y(v) {}
    tvec2(T x, T y): x(x), y(y) {}

    template<typename U>
    explicit tvec2(tvec2<U> v): x(v.x), y(v.y) {}

	template<typename U>
	explicit tvec2(tvec4<U> v) : x(v.x), y(v.y) {}

	template<typename U>
	explicit tvec2(tvec3<U> v) : x(v.x), y(v.y) {}

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

	const T&     operator[](int idx)const {return *(&x + idx);}
	      T&     operator[](int idx)      {return *(&x + idx);}

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
inline tvec4<T>::tvec4(const tvec3<T>& v, T w) : x(v.x), y(v.y), z(v.z), w(w) {}

template<typename T>
inline tvec4<T>::tvec4(T x, const tvec3<T>& v) : x(x), y(v.x), z(v.y), w(v.z) {}

template<typename T>
inline tvec4<T>::tvec4(const tvec2<T>& v1, const tvec2<T>& v2) : x(v1.x), y(v1.y), z(v2.x), w(v2.y) {}

template<typename T>
inline tvec4<T>::tvec4(const tvec2<T>& v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {}

template<typename T>
inline tvec4<T>::tvec4(T x, T y, const tvec2<T>& v) : x(x), y(y), z(v.x), w(v.y) {}

template<typename T>
inline tvec3<T>::tvec3(T x, const tvec2<T>& v) : x(x), y(v.x), z(v.y) {}

template<typename T>
inline tvec3<T>::tvec3(const tvec2<T>& v, T z) : x(v.x), y(v.y), z(z) {}

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

// n must be normalized
inline vec3 reflect(vec3 toReflect, vec3 n)
{
	return dot(toReflect, n) * n * -2.f + toReflect;
}

struct mat3
{
        mat3() = default;
        mat3(vec3 i, vec3 j, vec3 k):
            i(i),
            j(j),
            k(k)
    {}

	vec3 i = { 1.f, 0.f, 0.f };
	vec3 j = { 0.f, 1.f, 0.f };
	vec3 k = { 0.f, 0.f, 1.f };

	vec3& operator[](int idx) { return *(&i + idx); }
	const vec3& operator[](int idx) const { return *(&i + idx); }
};

inline vec3 operator*(const mat3& m, vec3 v)
{
	return v.x * m.i + v.y * m.j + v.z * m.k;
}

inline mat3 operator*(const mat3& lhs, const mat3& rhs)
{
	return
	mat3{
		lhs * rhs.i,
		lhs * rhs.j,
		lhs * rhs.k
	};
}

struct mat4
{
	vec4 i = { 1.f, 0.f, 0.f, 0.f };
	vec4 j = { 0.f, 1.f, 0.f, 0.f };
	vec4 k = { 0.f, 0.f, 1.f, 0.f };
	vec4 w = { 0.f, 0.f, 0.f, 1.f };
};

vec4 operator*(const mat4& m, vec4 v)
{
	return m.i * v.x + m.j * v.y + m.k * v.z + m.w * v.w;
}

mat4 operator*(const mat4& ml, const mat4& mr)
{
	mat4 m;
	m.i = ml * mr.i;
	m.j = ml * mr.j;
	m.k = ml * mr.k;
	m.w = ml * mr.w;
	return m;
}

mat4 translate(vec3 v)
{
	mat4 m;
	m.w = vec4(v, 1.f);
	return m;
}

mat4 scale(vec3 scale)
{
	mat4 m;
	m.i.x = scale.x;
	m.j.y = scale.y;
	m.k.z = scale.z;
	return m;
}

mat4 transpose(const mat4& m)
{
	mat4 mt;
	mt.i = { m.i.x, m.j.x, m.k.x, m.w.x };
	mt.j = { m.i.y, m.j.y, m.k.y, m.w.y };
	mt.k = { m.i.z, m.j.z, m.k.z, m.w.z };
	mt.w = { m.i.w, m.j.w, m.k.w, m.w.w };
	return mt;
}

mat4 lookAt(vec3 pos, vec3 target, vec3 up)
{
	vec3 k = -normalize(target - pos);
	vec3 i = normalize(cross(up, k));
	vec3 j = cross(k, i);

	mat4 basis;
	basis.i = vec4(i, 0.f);
	basis.j = vec4(j, 0.f);
	basis.k = vec4(k, 0.f);
	
	// change of a basis matrix is orthogonal (i, j, k, h are unit vectors and are perpendicular)
    // so inverse equals transpose (but I don't know the details)
	mat4 bInverse = transpose(basis);
	return bInverse * translate(-pos);
}

// windows...
#undef near
#undef far

// songho.ca/opengl/gl_projectionmatrix.html

mat4 frustrum(float right, float top, float near, float far)
{
	mat4 m;
	m.i = { near / right, 0.f, 0.f, 0.f };
	m.j = { 0.f, near / top, 0.f, 0.f };
	m.k = { 0.f, 0.f, -(far + near) / (far - near), -1.f};
	m.w = { 0.f, 0.f, (-2.f * far * near) / (far - near), 0.f };
	return m;
}

#define Pi 3.14159265359f

inline float toRadians(float degrees)
{
	return degrees / 360.f * 2.f * Pi;
}

// fovy is in degrees
mat4 perspective(float fovy, float aspect, float near, float far)
{
	fovy = toRadians(fovy);
	float top = tanf(fovy) * near;
	float right = aspect * top;
	return frustrum(right, top, near, far);
}

// todo: 3x3 matrices, rotation around any axis
// in degrees
mat4 rotateY(float angle)
{
	const float rad = toRadians(angle);
	const float sin = sinf(rad);
	const float cos = cosf(rad);
	mat4 m;
	m.i.x = cos;
	m.i.z = -sin;
	m.k.x = sin;
	m.k.z = cos;
	return m;
}

// in degrees
mat4 rotateX(float angle)
{
	const float rad = toRadians(angle);
	const float sin = sinf(rad);
	const float cos = cosf(rad);
	mat4 m;
	m.j.y = cos;
	m.j.z = sin;
	m.k.y = -sin;
	m.k.z = cos;
	return m;
}

// todo math stuff:
// * rotation
// * ortho
// * inverse
// * quaternions
// * frustrum culling

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

struct GLFWwindow;
GLFWwindow* gWindow;

// Rasterizer specific vvv

struct RGB8
{
	unsigned char r, g, b;
};

struct Index
{
	int position, texCoord, normal, tangent;
};

struct Face
{
	Index indices[3];
};

// 'Texture' is used for a gpu texture...
struct Bitmap
{
	vec3 sample(vec2 texCoord) 
	{
		assert(0.f <= texCoord.x && texCoord.x <= 1.f);
		assert(0.f <= texCoord.y && texCoord.y <= 1.f);

		if (!data)
			return {0.f, 1.f, 0.f};

		ivec2 pos = ivec2(texCoord * vec2(size - 1));
		RGB8 texel = data[pos.y * size.x + pos.x];
		return { texel.r / 255.f, texel.g / 255.f, texel.b / 255.f };
	}

	RGB8* data;
	ivec2 size;
};

void loadBitmapFromFile(Bitmap& bitmap, const char* filename);
void deleteBitmap(Bitmap& bitmap);

// todo: this is not cache friendly
struct Model
{
	Array<Face> faces;
	Array<vec3> positions;
	Array<vec2> texCoords;
	Array<vec3> normals;
	Array<vec3> tangents;
	Bitmap diffuseTexture;
	Bitmap specularTexture;
	Bitmap glowTexture;
	Bitmap tangentNormalMap;
};

struct Framebuffer
{
	Array<RGB8> colorArray;
	Array<float> depthArray;
	ivec2 size;
};

// execution: vertex + fragment, next face, vertex + fragment, ...
class Shader
{
public:
	virtual ~Shader() = default;
	virtual vec4 vertex(int faceIdx, int triangleVertexIdx) = 0;
	virtual vec3 fragment(vec3 barycentricCoords) = 0;

	struct
	{
		bool enable = false;
		vec4 positions[3][3];
	} tangentDebug;
};

class Shader1 : public Shader
{
public:
	vec4 vertex(int faceIdx, int triangleVertexIdx) override;
	vec3 fragment(vec3 barycentricCoords) override;

	Model* model;

	struct
	{
		mat4 model;
		mat4 view;
		mat4 projection;
	} mat;

	struct
	{
		vec2 texCoords[3];
		vec3 positions[3];
		mat3 TBN[3]; // tanget - bitangent - normal (tangent basis)
	} v; // varyings

	struct
	{
		vec3 dir; // will be normalized in fragment()
	} light;

	bool style2 = false;
	vec3 cameraPos;
	bool useNormalMap = true;
};

// NDC is left handed coordinate system (z points into the screen)
struct RenderCommand
{
	Framebuffer* fb;
	int numFraces;
	Shader* shader;
	bool wireframe = false;
	bool depthTest = true;
	bool cullFrontFaces = false;
	bool cullBackFaces = true;
};

// actually Camera's are a good fit for virtual functions (processEvent, update, imgui)
// and base members (pos, view)
class ArcballCamera
{
public:
	void processEvent(const WinEvent& event);
	// process events first
	void update();
	void imgui();

	float zoomSensitivity = 1.f; 
	float rotateSensitivity = 0.5f; // degrees / screen coordinates

	// get after update();
	mat4 view;
	vec3 pos = { 0.f, 0.f, 2.f };

private:
	vec2 cursorPos_;
	vec2 cursorPosDelta_ = vec2(0.f);
	bool buttonPressed_ = false;
	float scrollDelta_ = 0.f;
};

// fps camera
class Camera3d
{
public:
	Camera3d();

	void captureMouse(); // off on start
	void processEvent(const WinEvent& event);
	// process events first
	void update(float time);
	void imgui();

	bool forwardXZonly = false; // disable flying with W and S controls
	vec3 up = { 0.f, 1.f, 0.f }; // will be normalized in update()
	float speed = 2.f;
	float sensitivity = 0.1f; // degrees / screen coordinates
	                          // from GLFW - screen coordinates are not always pixels

	// get after update()

	mat4 view;
	vec3 pos = { 0.f, 0.2f, 1.2f };
	// degrees
	float pitch = 0.f;
	float yaw = 0.f;

private:
	enum
	{
		Forward,
		Back,
		Left,
		Right,
		Up,
		Down,
		ToggleMouseCapture,
		NumControls
	};

	int controls_[NumControls];

	struct
	{
		bool pressed[NumControls] = {};
		bool held[NumControls] = {};
	} keys_;

	vec2 cursorPos_;
	bool mouseCapture_ = false;
	bool firstCursorEvent_;

	bool cActive(int control) const { return keys_.pressed[control] || keys_.held[control];}
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

	// todo: make them static in render() (do not need cleanup)
	// todo: make one function execute(frame) instead of processInput(), update(), render()
	bool useFpsCamera_ = true;
	Shader1 shader_;
	RenderCommand rndCmd_;
	Camera3d camera_;
	ArcballCamera arcball_;
};
