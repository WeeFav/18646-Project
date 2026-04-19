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

void writeRasterData(const std::vector<RasterData>& data, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        const RasterData& r = data[i];

        for (int j = 0; j < 3; ++j) {
            // Header: t<i> v<j>
            out << "t" << i << " v" << j << "\n";

            // ndc (x y z w)
            out << r.ndc[j][0] << " "
                << r.ndc[j][1] << " "
                << r.ndc[j][2] << " "
                << r.ndc[j][3] << "\n";

            // screen (x y)
            out << r.screen[j][0] << " "
                << r.screen[j][1] << "\n";

            // normal (x y z)
            out << r.varying_nrm[j][0] << " "
                << r.varying_nrm[j][1] << " "
                << r.varying_nrm[j][2] << "\n";
        }
    }
}

struct PhongShader : IShader {
    const Model &model;
    vec3 l;
    vec3 tri[3];

    PhongShader(const vec3 light, const Model &m) : model(m) {
        l = normalized((ModelView * vec4{light.x, light.y, light.z, 0.}).xyz());
    }

    virtual vec4 vertex(const int face, const int vert, vec3 &varying_nrm) {
        vec3 v = model.vert(face, vert);
        vec3 n = model.normal(face, vert);
        varying_nrm = (ModelView.invert_transpose() * vec4{n.x, n.y, n.z, 0.}).xyz();
        vec4 gl_Position = ModelView * vec4{v.x, v.y, v.z, 1.};
        tri[vert] = gl_Position.xyz();
        return Perspective * gl_Position;
    }

    virtual std::pair<bool, TGAColor> fragment(const vec3 bar, const vec3 (&varying_nrm)[3]) const {
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

    // Results file for CPU baseline
    std::ofstream csv_file("results_baseline.csv");
    csv_file << "Resolution,Eye_Setting,Light_Setting,All_Transform_ms,Raster_Loop_ms,Total_ms\n";

    int resolutions[] = {16, 32, 64, 128};
    vec3 eye_settings[] = {{0, 1, 3}, {-3, 1, 0}, {3, 1, 0}, {0, 4, 0}, {2, 2, 2}};
    vec3 light_settings[] = {{0, 1, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}};
    vec3 center{0.065, 0.4725, 0};
    vec3 up{0, 1, 0};

    for (int res : resolutions) {
        // Create directory per resolution for baseline
        std::string dir_path = base_name + "_results_baseline/res_" + std::to_string(res);
        fs::create_directories(dir_path);

        int img_size = res * 128;
        
        // Load the decimated model corresponding to this resolution
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
                std::vector<RasterData> raster_data(model.nfaces());

                tsc_counter tt0, tt1;
                RDTSC(tt0);

                // --- Vertex Transform / Prepare Raster Data ---
                for (int f = 0; f < model.nfaces(); f++) {
                    Triangle clip;
                    for (int v = 0; v < 3; v++) {
                        clip[v] = shader.vertex(f, v, raster_data[f].varying_nrm[v]);
                    }
                    prepare_raster_data(clip, raster_data[f]);
                }
                RDTSC(tt1);

                // --- Rasterization ---
                tsc_counter rl0, rl1;
                RDTSC(rl0);
                for (int f = 0; f < model.nfaces(); f++) {
                    rasterize(raster_data[f], shader, framebuffer);
                }
                RDTSC(rl1);

                // Time calculation in milliseconds
                double transform_ms = COUNTER_DIFF(tt1, tt0, CYCLES) * 1000.0 / PROCESSOR_FREQ;
                double raster_loop_ms = COUNTER_DIFF(rl1, rl0, CYCLES) * 1000.0 / PROCESSOR_FREQ;
                double total_ms = transform_ms + raster_loop_ms;

                // Save metrics to CSV
                csv_file << res << ","
                         << eye.x << "_" << eye.y << "_" << eye.z << ","
                         << light.x << "_" << light.y << "_" << light.z << ","
                         << transform_ms << ","
                         << raster_loop_ms << ","
                         << total_ms << "\n";
                csv_file.flush();

                // Save TGA into the specific resolution baseline folder
                std::stringstream tga_ss;
                tga_ss << dir_path << "/cpu_out_e" << (int)eye.x << (int)eye.y << (int)eye.z << "_l" << (int)light.x << (int)light.y << (int)light.z << ".tga";
                framebuffer.write_tga_file(tga_ss.str().c_str());

                std::cout << "[CONFIG] Res: " << res 
                          << " | Eye: (" << eye.x << ", " << eye.y << ", " << eye.z << ")"
                          << " | Transform(ms): " << transform_ms 
                          << " | Raster(ms): " << raster_loop_ms 
                          << " | Total(ms): " << total_ms << std::endl;
            }
        }
    }

    csv_file.close();
    return 0;
}