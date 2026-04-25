#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#include "model.h"
#include "tgaimage.h"
#include "geometry.h"

namespace fs = std::filesystem;

// === Structs ===
typedef vec4 Triangle[3];
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

// Binning Pass 1: count how many triangles overlap each tile.
// One thread per triangle. Mirrors the ABC.det() < 1 cull from the CPU rasterizer
// (mat::det() is not __device__, so we compute the 3x3 determinant inline).
__global__ void binning_pass1(
    RasterData *d_raster_data,
    int        *d_tile_count,
    int         nfaces,
    int         num_tiles_x,
    int         num_tiles_y,
    int         tile_width,
    int         tile_height
) {
    // Each thread processes its triangles, atomicAdd to shared memory
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
        int tx_min = max(0,              (int)floor(bbminx / tile_width));
        int tx_max = min(num_tiles_x - 1,(int)floor(bbmaxx / tile_width));
        int ty_min = max(0,              (int)floor(bbminy / tile_height));
        int ty_max = min(num_tiles_y - 1,(int)floor(bbmaxy / tile_height));

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
    int         tile_width,
    int         tile_height
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

        int tx_min = max(0,              (int)floor(bbminx / tile_width));
        int tx_max = min(num_tiles_x - 1,(int)floor(bbmaxx / tile_width));
        int ty_min = max(0,              (int)floor(bbminy / tile_height));
        int ty_max = min(num_tiles_y - 1,(int)floor(bbmaxy / tile_height));

        for (int ty = ty_min; ty <= ty_max; ty++)
            for (int tx = tx_min; tx <= tx_max; tx++) {
                int pos = atomicAdd(&d_tile_cursor[ty * num_tiles_x + tx], 1);
                d_triangle_list[pos] = tri_id;
            }
    }
}

__global__ void init_raster_buffers(double *d_zbuffer, unsigned char *d_colorbuffer, int npixels) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < npixels;
         idx += gridDim.x * blockDim.x)
    {
        d_zbuffer[idx] = -1000.0;
        d_colorbuffer[idx * 3 + 0] = 0;
        d_colorbuffer[idx * 3 + 1] = 0;
        d_colorbuffer[idx * 3 + 2] = 0;
    }
}

// One block rasterizes one tile; each thread handles one pixel in that tile.
__global__ void rasterize_tiled(
    const RasterData *d_raster_data,
    const int        *d_tile_count,
    const int        *d_tile_start,
    const int        *d_triangle_list,
    int               num_tiles_x,
    int               tile_width,
    int               tile_height,
    int               width,
    int               height,
    vec3              light_cam,
    double           *d_zbuffer,
    unsigned char    *d_colorbuffer
) {
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    int tile_id = ty * num_tiles_x + tx;

    int px = tx * tile_width + threadIdx.x;
    int py = ty * tile_height + threadIdx.y;
    if (threadIdx.x >= tile_width || threadIdx.y >= tile_height) return;
    if (px >= width || py >= height) return;

    int pix_id = px + py * width;
    double best_z = d_zbuffer[pix_id];
    unsigned char best_color[3] = {d_colorbuffer[pix_id * 3 + 0], d_colorbuffer[pix_id * 3 + 1], d_colorbuffer[pix_id * 3 + 2]};

    int tri_count = d_tile_count[tile_id];
    int tri_begin = d_tile_start[tile_id];

    for (int i = 0; i < tri_count; ++i) {
        int tri_id = d_triangle_list[tri_begin + i];
        const RasterData &tri = d_raster_data[tri_id];

        const vec2 &a = tri.screen[0];
        const vec2 &b = tri.screen[1];
        const vec2 &c = tri.screen[2];

        double den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if (den <= 1e-12) continue;

        double alpha = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / den;
        double beta  = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / den;
        double gamma = 1.0 - alpha - beta;
        if (alpha < 0.0 || beta < 0.0 || gamma < 0.0) continue;

        double z = alpha * tri.ndc[0].z + beta * tri.ndc[1].z + gamma * tri.ndc[2].z;
        if (z <= best_z) continue;

        vec3 n = tri.varying_nrm[0] * alpha + tri.varying_nrm[1] * beta + tri.varying_nrm[2] * gamma;
        n = normalized(n);

        double nl = n * light_cam;
        double diff = fmax(0.0, nl);

        vec3 r = normalized(n * (2.0 * nl) - light_cam);
        double spec = pow(fmax(r.z, 0.0), 35.0);

        double intensity = fmin(1.0, 0.3 + 0.4 * diff + 0.9 * spec);
        std::uint8_t val = static_cast<std::uint8_t>(255.0 * intensity);

        best_z = z;
        best_color[0] = val;
        best_color[1] = val;
        best_color[2] = val;
    }

    d_zbuffer[pix_id] = best_z;
    d_colorbuffer[pix_id * 3 + 0] = best_color[0];
    d_colorbuffer[pix_id * 3 + 1] = best_color[1];
    d_colorbuffer[pix_id * 3 + 2] = best_color[2];
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

    vec2 tile_sizes[] = {{8,8},{8,16},{8,32},{16,8},{16,16},{16,32},{32,8},{32,16},{32,32}};

    for (vec2 tile_size: tile_sizes) {
        int TILE_WIDTH = tile_size.x;
        int TILE_HEIGHT = tile_size.y;

        std::string tile_tag = std::to_string(TILE_WIDTH) + "x" + std::to_string(TILE_HEIGHT);
        std::ofstream csv_file("results_" + tile_tag + ".csv");
        csv_file << "Resolution,Eye_Setting,Light_Setting,Transform_ms,Binning_ms,Raster_ms,Total_ms\n";

        std::cout << "Tile size: " << tile_tag << std::endl;      

        int resolutions[] = {16,32,64,128};
        vec3 eye_settings[] = {{0, 1, 3}, {-3, 1, 0}, {3, 1, 0}, {0, 4, 0}, {2, 2, 2}};
        vec3 light_settings[] = {{0, 1, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}};
        vec3 center{0.065, 0.4725, 0};
        vec3 up{0, 1, 0};

        for (int res : resolutions) {
            // std::string dir_path = base_name + "_results_tile_" + tile_tag + "/res_" + std::to_string(res);
            // fs::create_directories(dir_path);

            int img_size = res * 128;
            std::stringstream obj_ss;
            obj_ss << base_name << "_" << res << ".obj";

            Model model(obj_ss.str().c_str());
            if (model.nfaces() == 0) {
                std::cerr << "Model " << obj_ss.str() << " not found. Skipping resolution " << res << std::endl;
                continue;
            }

            // ALLOCATION HERE: Once per resolution to prevent leaks/noise
            RasterData *d_raster_data;
            vec3 *d_verts, *d_norms;
            int *d_facet_vrt, *d_facet_nrm;
            double *d_zbuffer;
            unsigned char *d_colorbuffer;
            int *d_tile_count, *d_tile_start;

            int num_pixels = img_size * img_size;
            int num_tiles_x = (img_size + TILE_WIDTH - 1) / TILE_WIDTH;
            int num_tiles_y = (img_size + TILE_HEIGHT - 1) / TILE_HEIGHT;
            int num_tiles   = num_tiles_x * num_tiles_y;

            cudaMalloc(&d_raster_data, model.nfaces() * sizeof(RasterData));
            cudaMalloc(&d_verts, model.nverts() * sizeof(vec3));
            cudaMalloc(&d_norms, model.norms.size() * sizeof(vec3));
            cudaMalloc(&d_facet_vrt, model.nfaces() * 3 * sizeof(int));
            cudaMalloc(&d_facet_nrm, model.nfaces() * 3 * sizeof(int));
            cudaMalloc(&d_zbuffer, num_pixels * sizeof(double));
            cudaMalloc(&d_colorbuffer, num_pixels * sizeof(unsigned char) * 3);
            cudaMalloc(&d_tile_count, num_tiles * sizeof(int));
            cudaMalloc(&d_tile_start, num_tiles * sizeof(int));

            for (vec3 eye : eye_settings) {
                for (vec3 light : light_settings) {
                    for (int k=0; k<10; k++) {
                        lookat(eye, center, up);
                        init_perspective(norm(eye - center));
                        init_viewport(img_size / 16, img_size / 16, img_size * 7 / 8, img_size * 7 / 8);

                        TGAImage framebuffer(img_size, img_size, TGAImage::RGB);
                        cudaEvent_t ev_tt0, ev_tt1, ev_tb0, ev_tb1, ev_rl0, ev_rl1;
                        cudaEventCreate(&ev_tt0); cudaEventCreate(&ev_tt1);
                        cudaEventCreate(&ev_tb0); cudaEventCreate(&ev_tb1);
                        cudaEventCreate(&ev_rl0); cudaEventCreate(&ev_rl1);

                        cudaEventRecord(ev_tt0);
                        // Vertex Transform
                        cudaMemcpy(d_verts, model.verts.data(), model.nverts() * sizeof(vec3), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_norms, model.norms.data(), model.norms.size() * sizeof(vec3), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_facet_vrt, model.facet_vrt.data(), model.nfaces() * 3 * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_facet_nrm, model.facet_nrm.data(), model.nfaces() * 3 * sizeof(int), cudaMemcpyHostToDevice);

                        cudaMemcpyToSymbol(d_ModelView, &ModelView, sizeof(mat<4,4>));
                        cudaMemcpyToSymbol(d_ModelViewInv, &ModelViewInv, sizeof(mat<4,4>));
                        cudaMemcpyToSymbol(d_Viewport, &Viewport, sizeof(mat<4,4>));
                        cudaMemcpyToSymbol(d_Perspective, &Perspective, sizeof(mat<4,4>));

                        int nblocks_bin = (model.nfaces() + 255) / 256;
                        vertex_transform<<<nblocks_bin, 256>>>(d_verts, d_norms, d_facet_vrt, d_facet_nrm, d_raster_data, model.nverts(), model.nfaces());
                        cudaEventRecord(ev_tt1);
                        cudaEventSynchronize(ev_tt1);


                        // Binning
                        cudaEventRecord(ev_tb0);
                        cudaMemset(d_tile_count, 0, num_tiles * sizeof(int));

                        binning_pass1<<<nblocks_bin, 256>>>(d_raster_data, d_tile_count, model.nfaces(), num_tiles_x, num_tiles_y, TILE_WIDTH, TILE_HEIGHT);
                        
                        thrust::exclusive_scan(thrust::device_ptr<int>(d_tile_count), thrust::device_ptr<int>(d_tile_count + num_tiles), thrust::device_ptr<int>(d_tile_start));
                        int total_overlaps = thrust::reduce(thrust::device_ptr<int>(d_tile_count), thrust::device_ptr<int>(d_tile_count + num_tiles));

                        int *d_triangle_list = nullptr;
                        if (total_overlaps > 0) {
                            cudaMalloc(&d_triangle_list, total_overlaps * sizeof(int));
                            int *d_tile_cursor;
                            cudaMalloc(&d_tile_cursor, num_tiles * sizeof(int));
                            cudaMemcpy(d_tile_cursor, d_tile_start, num_tiles * sizeof(int), cudaMemcpyDeviceToDevice);
                            binning_pass2<<<nblocks_bin, 256>>>(d_raster_data, d_tile_cursor, d_triangle_list, model.nfaces(), num_tiles_x, num_tiles_y, TILE_WIDTH, TILE_HEIGHT);
                            cudaFree(d_tile_cursor);
                        }
                        cudaEventRecord(ev_tb1);
                        cudaEventSynchronize(ev_tb1);


                        // Rasterization
                        cudaEventRecord(ev_rl0);
                        init_raster_buffers<<<(num_pixels + 255) / 256, 256>>>(d_zbuffer, d_colorbuffer, num_pixels);

                        if (total_overlaps > 0) {
                            vec3 light_cam = normalized((ModelView * vec4{light.x, light.y, light.z, 0.}).xyz());
                            rasterize_tiled<<<dim3(num_tiles_x, num_tiles_y), dim3(TILE_WIDTH, TILE_HEIGHT)>>>(d_raster_data, d_tile_count, d_tile_start, d_triangle_list, num_tiles_x, TILE_WIDTH, TILE_HEIGHT, img_size, img_size, light_cam, d_zbuffer, d_colorbuffer);
                        }
                        cudaMemcpy(framebuffer.data.data(), d_colorbuffer, num_pixels * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);
                        cudaEventRecord(ev_rl1);
                        cudaEventSynchronize(ev_rl1);

                        if (d_triangle_list) cudaFree(d_triangle_list);

                        // logging
                        float transform_ms = 0.0f, binning_ms = 0.0f, raster_ms = 0.0f;
                        cudaEventElapsedTime(&transform_ms, ev_tt0, ev_tt1);
                        cudaEventElapsedTime(&binning_ms, ev_tb0, ev_tb1);
                        cudaEventElapsedTime(&raster_ms, ev_rl0, ev_rl1);
                        float total_ms = transform_ms + binning_ms + raster_ms;

                        csv_file << res << "," << eye.x << "_" << eye.y << "_" << eye.z << "," << light.x << "_" << light.y << "_" << light.z << "," << transform_ms << "," << binning_ms << "," << raster_ms << "," << total_ms << "\n";
                        csv_file.flush();

                        // std::stringstream tga_ss;
                        // tga_ss << dir_path << "/gpu_out_e" << (int)eye.x << (int)eye.y << (int)eye.z << "_l" << (int)light.x << (int)light.y << (int)light.z << ".tga";
                        // framebuffer.write_tga_file(tga_ss.str().c_str());

                        std::cout << "[CONFIG] Res: " << res << " | Eye: (" << eye.x << ", " << eye.y << ", " << eye.z << ") | Light: (" << light.x << ", " << light.y << ", " << light.z << ") | Transform(ms): " << transform_ms << " | Binning(ms): " << binning_ms << " | Raster(ms): " << raster_ms << " | Total(ms): " << total_ms << std::endl;

                        cudaEventDestroy(ev_tt0); cudaEventDestroy(ev_tt1);
                        cudaEventDestroy(ev_tb0); cudaEventDestroy(ev_tb1);
                        cudaEventDestroy(ev_rl0); cudaEventDestroy(ev_rl1);
                    }
                }
            }
            // cleanup per resolution
            cudaFree(d_raster_data); cudaFree(d_verts); cudaFree(d_norms);
            cudaFree(d_facet_vrt); cudaFree(d_facet_nrm);
            cudaFree(d_zbuffer); cudaFree(d_colorbuffer);
            cudaFree(d_tile_count); cudaFree(d_tile_start);
        }

        csv_file.close();
    }

    return 0;
}
