#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem> // Required for directory creation
#include "model.h"
#include "tgaimage.h"
#include "geometry.h"
#include "rdtsc.h"

namespace fs = std::filesystem;

// === Structs ===
typedef vec4 Triangle[3]; // a triangle primitive is made of three ordered points
struct RasterData {
    vec4 ndc[3];
    vec2 screen[3];
    vec3 varying_nrm[3];
};

// === Global Variables ===
mat<4,4> ModelView, ModelViewInv, Viewport, Perspective; // "OpenGL" state matrices
std::vector<double> zbuffer;               // depth buffer

// === GPU Constant ===
__constant__ mat<4,4> d_ModelView, d_ModelViewInv, d_Viewport, d_Perspective;

// === Helper functions ====
void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 n = normalized(eye-center);
    vec3 l = normalized(cross(up,n));
    vec3 m = normalized(cross(n, l));
    ModelView = mat<4,4>{{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}} *
                mat<4,4>{{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
    ModelViewInv = ModelView.invert_transpose();
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

// === GPU Kernel ===
__global__ void vertex_transform(vec3 *d_verts, vec3 *d_norms, int *d_facet_vrt, int *d_facet_nrm, RasterData *d_raster_data, int nverts, int nfaces) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int iterations = nfaces / (gridDim.x * blockDim.x);
    int remainder = nfaces % (gridDim.x * blockDim.x);
    
    int start;
    if (id < remainder) {
        start = (iterations + 1) * id;
    }
    else {
        start = (iterations * id) + remainder;
    }

    for (int face_idx = start; face_idx < start + iterations + (id < remainder); face_idx++) {
        Triangle clip;
        for (int nthvert = 0; nthvert < 3; nthvert++) {
            vec3 v = d_verts[d_facet_vrt[face_idx * 3 + nthvert]];
            vec3 n = d_norms[d_facet_nrm[face_idx * 3 + nthvert]];
            d_raster_data[face_idx].varying_nrm[nthvert] = (d_ModelViewInv * vec4{n.x, n.y, n.z, 0.}).xyz(); // transform normal from world to camera
            vec4 gl_Position = d_ModelView * vec4{v.x, v.y, v.z, 1.}; // transform vertex from world to camera
            clip[nthvert] = d_Perspective * gl_Position; // apply perspective projection
            
            vec4 ndc = clip[nthvert] / clip[nthvert].w; // camera to NDC
            d_raster_data[face_idx].ndc[nthvert] = ndc;
            d_raster_data[face_idx].screen[nthvert] = (d_Viewport * ndc).xy(); // NDC to screen
        }
    }
}

// === Main ===
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
    csv_file << "Resolution,Eye_Setting,Light_Setting,All_Transform_Cycles,Raster_Loop_Cycles,Total_Cycles\n";

    int resolutions[] = {16};
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
                std::vector<RasterData> raster_data(model.nfaces());

                tsc_counter tt0, tt1;
                RDTSC(tt0);

                // === Vertex Transform ===
                vec3 *d_verts;
                vec3 *d_norms;
                int *d_facet_vrt;
                int *d_facet_nrm;
                RasterData *d_raster_data; // output

                cudaMalloc(&d_verts, model.nverts() * sizeof(vec3));
                cudaMalloc(&d_norms, model.norms.size() * sizeof(vec3));
                cudaMalloc(&d_facet_vrt, model.nfaces() * 3 * sizeof(int));
                cudaMalloc(&d_facet_nrm, model.nfaces() * 3 * sizeof(int));
                cudaMalloc(&d_raster_data, model.nfaces() * sizeof(RasterData));

                cudaMemcpy(d_verts, model.verts.data(), model.nverts() * sizeof(vec3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_norms, model.norms.data(), model.norms.size() * sizeof(vec3), cudaMemcpyHostToDevice);
                cudaMemcpy(d_facet_vrt, model.facet_vrt.data(), model.nfaces() * 3 * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_facet_nrm, model.facet_nrm.data(), model.nfaces() * 3 * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemset(d_raster_data, 0, model.nfaces() * sizeof(RasterData)); 
 
                cudaMemcpyToSymbol(d_ModelView, &ModelView, sizeof(mat<4,4>));
                cudaMemcpyToSymbol(d_ModelViewInv, &ModelViewInv, sizeof(mat<4,4>));
                cudaMemcpyToSymbol(d_Viewport, &Viewport, sizeof(mat<4,4>));
                cudaMemcpyToSymbol(d_Perspective, &Perspective, sizeof(mat<4,4>));

                vertex_transform<<<256, 256>>>(d_verts, d_norms, d_facet_vrt, d_facet_nrm, d_raster_data, model.nverts(), model.nfaces());

                cudaDeviceSynchronize();

                cudaMemcpy(raster_data.data(), d_raster_data, model.nfaces() * sizeof(RasterData), cudaMemcpyDeviceToHost);

                RDTSC(tt1);

                std::stringstream raster_data_ss;
                raster_data_ss << dir_path << "/raster_data_e" << (int)eye.x << (int)eye.y << (int)eye.z << "_l" << (int)light.x << (int)light.y << (int)light.z << ".txt";
                writeRasterData(raster_data, raster_data_ss.str().c_str());

                


                cudaFree(d_verts);
                cudaFree(d_norms);
                cudaFree(d_facet_vrt);
                cudaFree(d_facet_nrm);
                cudaFree(d_raster_data);

                tsc_counter rl0, rl1;
                RDTSC(rl0);
                RDTSC(rl1);
                
                long long transform_cycles = COUNTER_DIFF(tt1, tt0, CYCLES);
                long long raster_loop_cycles = COUNTER_DIFF(rl1, rl0, CYCLES);
                long long total_cycles = transform_cycles + raster_loop_cycles;
                
                // Save metrics to CSV
                csv_file << res << "," 
                         << eye.x << "_" << eye.y << "_" << eye.z << ","
                         << light.x << "_" << light.y << "_" << light.z << ","
                         << transform_cycles << ","
                         << raster_loop_cycles << ","
                         << total_cycles << "\n";
                csv_file.flush(); 
                
                // Save TGA into the specific resolution folder
                std::stringstream tga_ss;
                tga_ss << dir_path << "/out_e" << (int)eye.x << (int)eye.y << (int)eye.z << "_l" << (int)light.x << (int)light.y << (int)light.z << ".tga";
                framebuffer.write_tga_file(tga_ss.str().c_str());

                std::cout << "[CONFIG] Res: " << res 
                          << " | Eye: (" << eye.x << ", " << eye.y << ", " << eye.z << ")"
                          << " | Light: (" << light.x << ", " << light.y << ", " << light.z << ")" 
                          << " | All Transform Cycles: " << transform_cycles
                          << " | Raster Loop Cycles: " << raster_loop_cycles
                          << " | Total Cycles: " << total_cycles << std::endl;
            }
        }
    }

    csv_file.close();
    return 0;
}
