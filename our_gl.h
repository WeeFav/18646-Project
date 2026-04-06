#include "tgaimage.h"
#include "geometry.h"

void lookat(const vec3 eye, const vec3 center, const vec3 up);
void init_perspective(const double f);
void init_viewport(const int x, const int y, const int w, const int h);
void init_zbuffer(const int width, const int height);

struct IShader {
    virtual std::pair<bool,TGAColor> fragment(const vec3 bar, const vec3 (&varying_nrm)[3]) const = 0;
};

typedef vec4 Triangle[3]; // a triangle primitive is made of three ordered points
struct RasterData {
    vec4 ndc[3];
    vec2 screen[3];
    vec3 varying_nrm[3];
};

void prepare_raster_data(const Triangle &clip, RasterData &out);
void rasterize(const RasterData &data, const IShader &shader, TGAImage &framebuffer);

