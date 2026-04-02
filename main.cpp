#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem> // Required for directory creation
#include <algorithm>
#include "our_gl.h"
#include "model.h"
#include "rdtsc.h"

namespace fs = std::filesystem;

extern mat<4, 4> ModelView, Perspective;
extern std::vector<double> zbuffer;

struct PhongShader : IShader {
    const Model &model;
    vec3 l;
    vec3 tri[3];
    vec3 varying_nrm[3];

    PhongShader(const vec3 light, const Model &m) : model(m) {
        l = normalized((ModelView * vec4{light.x, light.y, light.z, 0.}).xyz());
    }

    virtual vec4 vertex(const int face, const int vert) {
        vec3 v = model.vert(face, vert);
        vec3 n = model.normal(face, vert);
        varying_nrm[vert] = (ModelView.invert_transpose() * vec4{n.x, n.y, n.z, 0.}).xyz();
        vec4 gl_Position = ModelView * vec4{v.x, v.y, v.z, 1.};
        tri[vert] = gl_Position.xyz();
        return Perspective * gl_Position;
    }

    virtual std::pair<bool, TGAColor> fragment(const vec3 bar) const {
        TGAColor gl_FragColor = {255, 255, 255, 255};
        vec3 n = normalized(varying_nrm[0] * bar[0] + varying_nrm[1] * bar[1] + varying_nrm[2] * bar[2]);
        vec3 r = normalized(n * (n * l) * 2 - l);
        double ambient = .3;
        double diff = std::max(0., n * l);
        double spec = std::pow(std::max(r.z, 0.), 35);
        for (int channel : {0, 1, 2})
            gl_FragColor[channel] *= std::min(1., ambient + .4 * diff + .9 * spec);
        return {false, gl_FragColor};
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " base_name" << std::endl;
        return 1;
    }

    std::string base_name = argv[1];
    if (base_name.size() > 4 && base_name.substr(base_name.size() - 4) == ".obj") {
        base_name = base_name.substr(0, base_name.size() - 4);
    }

    // Results file for Python plotting
    std::ofstream csv_file("results.csv");
    csv_file << "Resolution,Eye_Setting,Light_Setting,Cycles\n";

    int resolutions[] = {16, 32, 64, 128};
    vec3 eye_settings[] = {{0, 1, 3}, {-3, 1, 0}, {3, 1, 0}, {0, 4, 0}, {2, 2, 2}};
    vec3 light_settings[] = {{0, 1, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}};
    vec3 center{0.065, 0.4725, 0};
    vec3 up{0, 1, 0};

    for (int res : resolutions) {
        // Define the output directory path
        std::string dir_path = base_name + "_results/res_" + std::to_string(res);
        fs::create_directories(dir_path);

        int img_size = res * 128; 
        
        // UPDATED: This now uses the full path provided in the command line argument
        std::stringstream obj_ss;
        obj_ss << base_name << "_" << res << ".obj"; 
        
        Model model(obj_ss.str().c_str());
        if (model.nfaces() == 0) {
            std::cerr << "Model " << obj_ss.str() << " not found. Skipping resolution " << res << std::endl;
            continue;
        }

        for (vec3 eye : eye_settings) {
            for (vec3 light : light_settings) {
                // Initialize scene state
                lookat(eye, center, up);
                init_perspective(norm(eye - center));
                init_viewport(img_size / 16, img_size / 16, img_size * 7 / 8, img_size * 7 / 8);
                init_zbuffer(img_size, img_size);
                
                TGAImage framebuffer(img_size, img_size, TGAImage::RGB);
                PhongShader shader(light, model);

                // Timing block using RDTSC 
                tsc_counter t0, t1;
                RDTSC(t0);
                for (int f = 0; f < model.nfaces(); f++) {
                    Triangle clip = {shader.vertex(f, 0), shader.vertex(f, 1), shader.vertex(f, 2)};
                    rasterize(clip, shader, framebuffer);
                }
                RDTSC(t1);

                long long cycles = COUNTER_DIFF(t1, t0, CYCLES);

                // Save metrics to CSV
                csv_file << res << "," 
                         << eye.x << "_" << eye.y << "_" << eye.z << ","
                         << light.x << "_" << light.y << "_" << light.z << ","
                         << cycles << "\n";
                csv_file.flush(); 

                // Save TGA into the specific resolution folder
                std::stringstream tga_ss;
                tga_ss << dir_path << "/out_e" << (int)eye.x << "_l" << (int)light.x << ".tga";
                framebuffer.write_tga_file(tga_ss.str().c_str());

                std::cout << "[CONFIG] Res: " << res 
                          << " | Eye: (" << eye.x << ", " << eye.y << ", " << eye.z << ")"
                          << " | Light: (" << light.x << ", " << light.y << ", " << light.z << ")" 
                          << " | Cycles: " << cycles << std::endl;
            }
        }
    }
    csv_file.close();
    return 0;
}