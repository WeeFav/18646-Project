#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "model.h"
#include "tgaimage.h"
#include "geometry.h"
#include "rdtsc.h"

namespace fs = std::filesystem;

#define TILE_SIZE 16

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

// === GPU Kernels ===

__global__ void vertex_transform(vec3 *d_verts, vec3 *d_norms, int *d_facet_vrt, int *d_facet_nrm, RasterData *d_raster_data, int nverts, int nfaces) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int face_idx = id; face_idx < nfaces; face_idx += gridDim.x * blockDim.x) {
        Triangle clip;
        for (int nthvert = 0; nthvert < 3; nthvert++) {
            vec3 v = d_verts[d_facet_vrt[face_idx * 3 + nthvert]];
            vec3 n = d_norms[d_facet_nrm[face_idx * 3 + nthvert]];
            d_raster_data[face_idx].varying_nrm[nthvert] = (d_ModelViewInv * vec4{n.x, n.y, n.z, 0.}).xyz(); // transform normal from world to camera
            vec4 gl_Position = d_ModelView * vec4{v.x, v.y, v.z, 1.}; // transform vertex from world to camera
            clip[nthvert] = d_Perspective * gl_Position; // apply perspective projection

            d_raster_data[face_idx].ndc[nthvert] = clip[nthvert] / clip[nthvert].w; // camera to NDC
            d_raster_data[face_idx].screen[nthvert] = (d_Viewport * d_raster_data[face_idx].ndc[nthvert]).xy(); // NDC to screen
        }
    }
}

// Binning Pass 1: count how many triangles overlap each tile.
// One thread per triangle. Mirrors the ABC.det() < 1 cull from the CPU rasterizer
// (mat::det() is not __device__, so we compute the 3x3 determinant inline).
__global__ void binning_pass1(
    RasterData *d_raster_data,
    int        *d_tile_count,
    int         nfaces,
    int         num_tiles_x,
    int         num_tiles_y,
    int         tile_size
) {
    for (int tri_id = blockIdx.x * blockDim.x + threadIdx.x;
         tri_id < nfaces;
         tri_id += gridDim.x * blockDim.x)
    {
        const vec2 *s = d_raster_data[tri_id].screen;

        // Signed-area det of ABC (same formula as our_gl.cpp: ABC.det() < 1)
        // Rows of ABC are {s[i].x, s[i].y, 1}; expanding the 3x3 determinant:
        double det = s[0].x * (s[1].y - s[2].y)
                   + s[1].x * (s[2].y - s[0].y)
                   + s[2].x * (s[0].y - s[1].y);
        if (det < 1.0) continue;

        // Screen-space bounding box
        double bbminx = fmin(s[0].x, fmin(s[1].x, s[2].x));
        double bbmaxx = fmax(s[0].x, fmax(s[1].x, s[2].x));
        double bbminy = fmin(s[0].y, fmin(s[1].y, s[2].y));
        double bbmaxy = fmax(s[0].y, fmax(s[1].y, s[2].y));

        // Tile range, clamped to valid tile indices
        int tx_min = max(0,              (int)floor(bbminx / tile_size));
        int tx_max = min(num_tiles_x - 1,(int)floor(bbmaxx / tile_size));
        int ty_min = max(0,              (int)floor(bbminy / tile_size));
        int ty_max = min(num_tiles_y - 1,(int)floor(bbmaxy / tile_size));

        for (int ty = ty_min; ty <= ty_max; ty++)
            for (int tx = tx_min; tx <= tx_max; tx++)
                atomicAdd(&d_tile_count[ty * num_tiles_x + tx], 1);
    }
}

// Binning Pass 2: scatter triangle IDs into the flat triangle_list.
// d_tile_cursor is a mutable copy of d_tile_start; atomicAdd on it gives each
// thread a unique slot without overwriting d_tile_start (which the rasterization
// kernel reads to locate each tile's list).
__global__ void binning_pass2(
    RasterData *d_raster_data,
    int        *d_tile_cursor,
    int        *d_triangle_list,
    int         nfaces,
    int         num_tiles_x,
    int         num_tiles_y,
    int         tile_size
) {
    for (int tri_id = blockIdx.x * blockDim.x + threadIdx.x;
         tri_id < nfaces;
         tri_id += gridDim.x * blockDim.x)
    {
        const vec2 *s = d_raster_data[tri_id].screen;

        double det = s[0].x * (s[1].y - s[2].y)
                   + s[1].x * (s[2].y - s[0].y)
                   + s[2].x * (s[0].y - s[1].y);
        if (det < 1.0) continue;

        double bbminx = fmin(s[0].x, fmin(s[1].x, s[2].x));
        double bbmaxx = fmax(s[0].x, fmax(s[1].x, s[2].x));
        double bbminy = fmin(s[0].y, fmin(s[1].y, s[2].y));
        double bbmaxy = fmax(s[0].y, fmax(s[1].y, s[2].y));

        int tx_min = max(0,              (int)floor(bbminx / tile_size));
        int tx_max = min(num_tiles_x - 1,(int)floor(bbmaxx / tile_size));
        int ty_min = max(0,              (int)floor(bbminy / tile_size));
        int ty_max = min(num_tiles_y - 1,(int)floor(bbmaxy / tile_size));

        for (int ty = ty_min; ty <= ty_max; ty++)
            for (int tx = tx_min; tx <= tx_max; tx++) {
                int pos = atomicAdd(&d_tile_cursor[ty * num_tiles_x + tx], 1);
                d_triangle_list[pos] = tri_id;
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
    csv_file << "Resolution,Eye_Setting,Light_Setting,All_Transform_Cycles,Binning_Cycles,Raster_Loop_Cycles,Total_Cycles\n";

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

                // Vertex inputs no longer needed; free them before binning
                cudaFree(d_verts);
                cudaFree(d_norms);
                cudaFree(d_facet_vrt);
                cudaFree(d_facet_nrm);

                cudaMemcpy(raster_data.data(), d_raster_data, model.nfaces() * sizeof(RasterData), cudaMemcpyDeviceToHost);

                RDTSC(tt1);

                std::stringstream raster_data_ss;
                raster_data_ss << dir_path << "/raster_data_e" << (int)eye.x << (int)eye.y << (int)eye.z << "_l" << (int)light.x << (int)light.y << (int)light.z << ".txt";
                writeRasterData(raster_data, raster_data_ss.str().c_str());

                // === Tiled Binning ===
                tsc_counter tb0, tb1;
                RDTSC(tb0);

                int num_tiles_x = img_size / TILE_SIZE;
                int num_tiles_y = img_size / TILE_SIZE;
                int num_tiles   = num_tiles_x * num_tiles_y;

                // Allocate and zero-initialize per-tile triangle counts
                int *d_tile_count;
                cudaMalloc(&d_tile_count, num_tiles * sizeof(int));
                cudaMemset(d_tile_count, 0, num_tiles * sizeof(int));

                // Pass 1: count triangle-tile overlaps
                int nblocks_bin = (model.nfaces() + 255) / 256;
                binning_pass1<<<nblocks_bin, 256>>>(
                    d_raster_data, d_tile_count,
                    model.nfaces(), num_tiles_x, num_tiles_y, TILE_SIZE);
                cudaDeviceSynchronize();

                // Prefix sum: tile_count[] -> tile_start[] (exclusive scan)
                int *d_tile_start;
                cudaMalloc(&d_tile_start, num_tiles * sizeof(int));
                thrust::exclusive_scan(
                    thrust::device_ptr<int>(d_tile_count),
                    thrust::device_ptr<int>(d_tile_count + num_tiles),
                    thrust::device_ptr<int>(d_tile_start));

                // Total triangle-tile overlap entries needed for triangle_list
                int total_overlaps = thrust::reduce(
                    thrust::device_ptr<int>(d_tile_count),
                    thrust::device_ptr<int>(d_tile_count + num_tiles));

                // Pass 2: scatter triangle IDs into flat triangle_list
                int *d_triangle_list = nullptr;
                int *d_tile_cursor   = nullptr;
                if (total_overlaps > 0) {
                    cudaMalloc(&d_triangle_list, total_overlaps * sizeof(int));

                    // d_tile_cursor is a mutable working copy of d_tile_start;
                    // Pass 2 atomicAdd's into it to assign unique slots per tile.
                    // d_tile_start is preserved unchanged for the rasterization kernel.
                    cudaMalloc(&d_tile_cursor, num_tiles * sizeof(int));
                    cudaMemcpy(d_tile_cursor, d_tile_start,
                               num_tiles * sizeof(int), cudaMemcpyDeviceToDevice);

                    binning_pass2<<<nblocks_bin, 256>>>(
                        d_raster_data, d_tile_cursor, d_triangle_list,
                        model.nfaces(), num_tiles_x, num_tiles_y, TILE_SIZE);
                    cudaDeviceSynchronize();
                }

                RDTSC(tb1);

                // d_raster_data no longer needed after binning
                cudaFree(d_raster_data);
                if (d_tile_cursor) cudaFree(d_tile_cursor);

                // section 2.3: pass d_tile_count, d_tile_start, d_triangle_list to the
                //       rasterization kernel (free first)
                cudaFree(d_tile_count);
                cudaFree(d_tile_start);
                if (d_triangle_list) cudaFree(d_triangle_list);

                // === Rasterization ===
                tsc_counter rl0, rl1;
                RDTSC(rl0);
                RDTSC(rl1);

                long long transform_cycles   = COUNTER_DIFF(tt1, tt0, CYCLES);
                long long binning_cycles     = COUNTER_DIFF(tb1, tb0, CYCLES);
                long long raster_loop_cycles = COUNTER_DIFF(rl1, rl0, CYCLES);
                long long total_cycles = transform_cycles + binning_cycles + raster_loop_cycles;

                // Save metrics to CSV
                csv_file << res << ","
                         << eye.x << "_" << eye.y << "_" << eye.z << ","
                         << light.x << "_" << light.y << "_" << light.z << ","
                         << transform_cycles << ","
                         << binning_cycles << ","
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
                          << " | Transform Cycles: " << transform_cycles
                          << " | Binning Cycles: " << binning_cycles
                          << " | Total Overlaps: " << total_overlaps
                          << " | Raster Loop Cycles: " << raster_loop_cycles
                          << " | Total Cycles: " << total_cycles << std::endl;
            }
        }
    }

    csv_file.close();
    return 0;
}
