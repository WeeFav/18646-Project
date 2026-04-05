#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem> // Required for directory creation
#include "model.h"

namespace fs = std::filesystem;

mat<4,4> ModelView, Viewport, Perspective; // "OpenGL" state matrices
std::vector<double> zbuffer;               // depth buffer

void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 n = normalized(eye-center);
    vec3 l = normalized(cross(up,n));
    vec3 m = normalized(cross(n, l));
    ModelView = mat<4,4>{{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}} *
                mat<4,4>{{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
}

void init_perspective(const double f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1/f,1}}};
}

void init_viewport(const int x, const int y, const int w, const int h) {
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void init_zbuffer(const int width, const int height) {
    zbuffer = std::vector(width*height, -1000.);
}

void prepare_raster_data(const Triangle &clip, RasterData &out) {
    out.ndc[0] = clip[0] / clip[0].w;
    out.ndc[1] = clip[1] / clip[1].w;
    out.ndc[2] = clip[2] / clip[2].w;
    out.screen[0] = (Viewport * out.ndc[0]).xy();
    out.screen[1] = (Viewport * out.ndc[1]).xy();
    out.screen[2] = (Viewport * out.ndc[2]).xy();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " base_name" << std::endl;
        return 1;
    }

    std::string base_name = argv[1];
    if (base_name.size() > 4 && base_name.substr(base_name.size() - 4) == ".obj") {
        base_name = base_name.substr(0, base_name.size() - 4);
    }

    int res = 16;
    vec3 eye = {0, 1, 3};
    vec3 light = {0, 1, 1};
    vec3 center{0.065, 0.4725, 0};
    vec3 up{0, 1, 0};

    std::string dir_path = base_name + "_results/res_" + std::to_string(res);
    fs::create_directories(dir_path);

    int img_size = res * 128; 
    
    // UPDATED: This now uses the full path provided in the command line argument
    std::stringstream obj_ss;
    obj_ss << base_name << "_" << res << ".obj"; 
    
    Model model(obj_ss.str().c_str());
    if (model.nfaces() == 0) {
        std::cerr << "Model " << obj_ss.str() << " not found. Skipping resolution " << res << std::endl;
        return 1;
    }

    // Initialize scene state
    lookat(eye, center, up);
    init_perspective(norm(eye - center));
    init_viewport(img_size / 16, img_size / 16, img_size * 7 / 8, img_size * 7 / 8);
    init_zbuffer(img_size, img_size);


}