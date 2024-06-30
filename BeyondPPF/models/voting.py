import cupy as cp


ppf_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting(
        const float *points, const float *outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }
''', 'ppf_voting')


ppf_retrieval_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting_retrieval(
        const float *point_pairs, const float *outputs, const float *probs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            float3 a = make_float3(point_pairs[idx * 6], point_pairs[idx * 6 + 1], point_pairs[idx * 6 + 2]);
            float3 b = make_float3(point_pairs[idx * 6 + 3], point_pairs[idx * 6 + 4], point_pairs[idx * 6 + 5]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }
''', 'ppf_voting_retrieval')


nocs_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void nocs_voting(
        const float *points, const float *nocs_norm, float *grid_obj, const float *corner, const float *sph_offsets, float res,
        int n_offsets, int n_samples, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_samples) {
            float l = nocs_norm[idx];
            float3 point = make_float3(points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]);
            
            // printf("point: %f, %f, %f\n", point.x, point.y, point.z);
            // printf("corner: %f, %f, %f\n", corner[0], corner[1], corner[2]);
            for (int i = 0; i < n_offsets; i++) {
                float3 offset = l * make_float3(sph_offsets[i * 3], sph_offsets[i * 3 + 1], sph_offsets[i * 3 + 2]);
                float3 center_grid = (point + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh);
            }
        }
    }
''', 'nocs_voting')


nocs_pair_voting_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    
    extern "C" __global__
    void nocs_pair_voting(
        const float *points, 
        const float *preds, 
        float *zyx_grid, 
        float *zyx_delta, 
        int *point_idxs, 
        int num_bins,
        int n_ppfs,
        int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float3 a = make_float3(preds[idx * 3], preds[idx * 3 + 1], preds[idx * 3 + 2]);
            
            int idx1 = point_idxs[idx * 2], idx2 = point_idxs[idx * 2 + 1];
            float3 b = make_float3(points[idx2 * 3], points[idx2 * 3 + 1], points[idx2 * 3 + 2]) \
                - make_float3(points[idx1 * 3], points[idx1 * 3 + 1], points[idx1 * 3 + 2]);
                
            a /= (length(a) + 1e-9);
            b /= (length(b) + 1e-9);
            
            float adotb = dot(a, b);
            float3 a2b[3];
            if (adotb > 1 - 1e-7) {
                a2b[0] = make_float3(1, 0, 0);
                a2b[1] = make_float3(0, 1, 0);
                a2b[2] = make_float3(0, 0, 1);
            } else {
                float3 c = cross(a, b);
                float c_norm_sq = dot(c, c);
                float d = (1 - adotb) / (c_norm_sq + 1e-9);
                a2b[0] = make_float3(1, 0, 0) + make_float3(0, -c.z, c.y) + d * make_float3(-c.y * c.y - c.z * c.z, c.x * c.y, c.x * c.z);
                a2b[1] = make_float3(0, 1, 0) + make_float3(c.z, 0, -c.x) + d * make_float3(c.x * c.y, -c.x * c.x - c.z * c.z, c.y * c.z);
                a2b[2] = make_float3(0, 0, 1) + make_float3(-c.y, c.x, 0) + d * make_float3(c.x * c.z, c.y * c.z, -c.x * c.x - c.y * c.y);
            }
            
            float N1 = b.x, N2 = b.y, N3 = b.z;
            
            for (int m = 0; m < n_rots; m++) {
                float theta = m * 2 * M_PI / n_rots;
                float r11 = a2b[0].x*((cos(theta) - 1)*N2*N2 + (cos(theta) - 1)*N3*N3 + 1) 
                    - a2b[1].x*(N3*sin(theta) + N1*N2*(cos(theta) - 1))
                    + a2b[2].x*(N2*sin(theta) - N1*N3*(cos(theta) - 1));
                float r21 = a2b[1].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N3*N3 + 1) 
                    + a2b[0].x*(N3*sin(theta) - N1*N2*(cos(theta) - 1)) 
                    - a2b[2].x*(N1*sin(theta) + N2*N3*(cos(theta) - 1));
                float r31 = a2b[2].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].x*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].x*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                float r32 = a2b[2].y*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].y*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].y*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                float r33 = a2b[2].z*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].z*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].z*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                
                float th = -asin(r31);
                float thcos = cos(th);
                float psi = atan2(r32 / thcos, r33 / thcos);
                float phi = atan2(r21 / thcos, r11 / thcos);
                if (isnan(th) || isnan(psi) || isnan(phi)) return;
                
                const int err_bound = 0;
                for (int i = -err_bound; i < err_bound + 1; i++)
                {
                    for (int j = -err_bound; j < err_bound + 1; j++)
                    {
                        for (int k = -err_bound; k < err_bound + 1; k++)
                        {
                            // int th_idx = clamp(int((th + M_PI / 2.f) / M_PI * num_bins) + i, 0, num_bins - 1);
                            int th_idx = clamp(int((r31 + 1.f) / 2.f * num_bins) + i, 0, num_bins - 1);
                            int psi_idx = clamp(int((psi + M_PI) / 2.f / M_PI * num_bins) + j, 0, num_bins - 1);
                            int phi_idx = clamp(int((phi + M_PI) / 2.f / M_PI * num_bins) + k, 0, num_bins - 1);
                            
                            float phi_delta = phi + M_PI - (M_PI / num_bins + phi_idx * 2.f * M_PI / num_bins);
                            float psi_delta = psi + M_PI - (M_PI / num_bins + psi_idx * 2.f * M_PI / num_bins);
                            float th_delta = r31 + 1.f - (1.f / num_bins + float(th_idx * 2.f) / num_bins);
                            
                            int idx = phi_idx * num_bins * num_bins + th_idx * num_bins + psi_idx;
                            
                            atomicAdd(&zyx_grid[idx], 1.f);
                            atomicAdd(&zyx_delta[idx * 3], phi_delta);
                            atomicAdd(&zyx_delta[idx * 3 + 1], th_delta);
                            atomicAdd(&zyx_delta[idx * 3 + 2], psi_delta);
                        }
                    }
                }
            }
        }
    }
''', 'nocs_pair_voting')

nocs_rot_voting_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    
    extern "C" __global__
    void nocs_rot_voting(
        const float *points, const float *nocs, const float *center, float *zyx_grid, float *zyx_delta, int *idx2s, int n_samples, int num_bins
    ) {
        const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
        int err_bound = 1;
        if (idx1 < n_samples) {
            // int idx2 = idx1 + n_samples;
            int idx2 = idx2s[idx1];
            if (length(make_float3(points[idx1 * 3], points[idx1 * 3 + 1], points[idx1 * 3 + 2]) 
                     - make_float3(points[idx2 * 3], points[idx2 * 3 + 1], points[idx2 * 3 + 2])) < 1e-2)
                return;
            
            float3 cent = *((const float3 *)center);
            float3 a = make_float3(nocs[idx1 * 3], nocs[idx1 * 3 + 1], nocs[idx1 * 3 + 2]);
            // float len_a = length(a);
            float3 b = make_float3(points[idx1 * 3], points[idx1 * 3 + 1], points[idx1 * 3 + 2]) - cent;
            // if (abs(length(a) - length(b)) > 1e-2) return;
            a /= (length(a) + 1e-9);
            b /= (length(b) + 1e-9);
            
            float adotb = dot(a, b);
            float3 a2b[3];
            if (adotb > 1 - 1e-7) {
                a2b[0] = make_float3(1, 0, 0);
                a2b[1] = make_float3(0, 1, 0);
                a2b[2] = make_float3(0, 0, 1);
            } else {
                float3 c = cross(a, b);
                float c_norm_sq = dot(c, c);
                float d = (1 - adotb) / (c_norm_sq + 1e-9);
                a2b[0] = make_float3(1, 0, 0) + make_float3(0, -c.z, c.y) + d * make_float3(-c.y * c.y - c.z * c.z, c.x * c.y, c.x * c.z);
                a2b[1] = make_float3(0, 1, 0) + make_float3(c.z, 0, -c.x) + d * make_float3(c.x * c.y, -c.x * c.x - c.z * c.z, c.y * c.z);
                a2b[2] = make_float3(0, 0, 1) + make_float3(-c.y, c.x, 0) + d * make_float3(c.x * c.z, c.y * c.z, -c.x * c.x - c.y * c.y);
            }
            
            // cent = make_float3(points[idx1 * 3], points[idx1 * 3 + 1], points[idx1 * 3 + 2]) - b * len_a;
            
            float3 y = make_float3(points[idx2 * 3], points[idx2 * 3 + 1], points[idx2 * 3 + 2]) - cent;
            y /= (length(y) + 1e-9);
            float3 e = make_float3(nocs[idx2 * 3], nocs[idx2 * 3 + 1], nocs[idx2 * 3 + 2]);
            e /= (length(e) + 1e-9);
            float3 x = make_float3(dot(a2b[0], e), dot(a2b[1], e), dot(a2b[2], e));
            
            // if (abs(length(y) - length(e)) > 1e-2) return;
            
            float y1 = y.x, y2 = y.y, y3 = y.z;
            float x1 = x.x, x2 = x.y, x3 = x.z;
            float N1 = b.x, N2 = b.y, N3 = b.z;
            float sin_coeff = (y2*(N1*(N1*x2 - N2*x1) - N3*(N2*x3 - N3*x2)) - y1*(N2*(N1*x2 - N2*x1) + N3*(N1*x3 - N3*x1)) + y3*(N1*(N1*x3 - N3*x1) + N2*(N2*x3 - N3*x2)));
            float cos_coeff = (y1*(N2*(N1*N3*x1 - x3*(N1*N1 + N2*N2) + N2*N3*x2) - N3*(N1*N2*x1 - x2*(N1*N1 + N3*N3) + N2*N3*x3)) - y2*(N1*(N1*N3*x1 - x3*(N1*N1 + N2*N2) + N2*N3*x2) - N3*(N1*N2*x2 - x1*(N2*N2 + N3*N3) + N1*N3*x3)) + y3*(N1*(N1*N2*x1 - x2*(N1*N1 + N3*N3) + N2*N3*x3) - N2*(N1*N2*x2 - x1*(N2*N2 + N3*N3) + N1*N3*x3)));
            float cc = y2*(N1*(N1*N3*x1 - x3*(N1*N1 + N2*N2 - 1) + N2*N3*x2) - N3*(N1*N2*x2 - x1*(N2*N2 + N3*N3 - 1) + N1*N3*x3)) - y1*(N2*(N1*N3*x1 - x3*(N1*N1 + N2*N2 - 1) + N2*N3*x2) - N3*(N1*N2*x1 - x2*(N1*N1 + N3*N3 - 1) + N2*N3*x3)) - y3*(N1*(N1*N2*x1 - x2*(N1*N1 + N3*N3 - 1) + N2*N3*x3) - N2*(N1*N2*x2 - x1*(N2*N2 + N3*N3 - 1) + N1*N3*x3));
            
            float len = length(make_float2(sin_coeff, cos_coeff));
            float alpha = atan2(sin_coeff / len, cos_coeff / len);
            float theta = 0;
            float cands[2];
            float dists[2];
            for (int i = 0; i < 2; i++)
            {
                float cand = (i * 2 - 1) * acos(-cc / len) + alpha;
                float3 rot[3] = {
                    make_float3((cos(cand) - 1)*N2*N2 + (cos(cand) - 1)*N3*N3 + 1,            - N3*sin(cand) - N1*N2*(cos(cand) - 1),              N2*sin(cand) - N1*N3*(cos(cand) - 1)),
                    make_float3(             N3*sin(cand) - N1*N2*(cos(cand) - 1), (cos(cand) - 1)*N1*N1 + (cos(cand) - 1)*N3*N3 + 1,            - N1*sin(cand) - N2*N3*(cos(cand) - 1)),
                    make_float3(           - N2*sin(cand) - N1*N3*(cos(cand) - 1),              N1*sin(cand) - N2*N3*(cos(cand) - 1), (cos(cand) - 1)*N1*N1 + (cos(cand) - 1)*N2*N2 + 1)
                };
                dists[i] = length(y - make_float3(dot(rot[0], x), dot(rot[1], x), dot(rot[2], x)));
                cands[i] = cand;
            }
            if (dists[0] < dists[1]) theta = cands[0];
            else theta = cands[1];

            float r11 = a2b[0].x*((cos(theta) - 1)*N2*N2 + (cos(theta) - 1)*N3*N3 + 1) 
                - a2b[1].x*(N3*sin(theta) + N1*N2*(cos(theta) - 1))
                + a2b[2].x*(N2*sin(theta) - N1*N3*(cos(theta) - 1));
            float r21 = a2b[1].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N3*N3 + 1) 
                + a2b[0].x*(N3*sin(theta) - N1*N2*(cos(theta) - 1)) 
                - a2b[2].x*(N1*sin(theta) + N2*N3*(cos(theta) - 1));
            float r31 = a2b[2].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                - a2b[0].x*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                + a2b[1].x*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
            float r32 = a2b[2].y*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                - a2b[0].y*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                + a2b[1].y*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
            float r33 = a2b[2].z*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                - a2b[0].z*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                + a2b[1].z*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
            
            float th = -asin(r31);
            float thcos = cos(th);
            float psi = atan2(r32 / thcos, r33 / thcos);
            float phi = atan2(r21 / thcos, r11 / thcos);
            if (isnan(th) || isnan(psi) || isnan(phi)) return;
            
            for (int i = -err_bound; i < err_bound + 1; i++)
            {
                for (int j = -err_bound; j < err_bound + 1; j++)
                {
                    for (int k = -err_bound; k < err_bound + 1; k++)
                    {
                        // int th_idx = clamp(int((th + M_PI / 2.f) / M_PI * num_bins) + i, 0, num_bins - 1);
                        int th_idx = clamp(int((r31 + 1.f) / 2.f * num_bins) + i, 0, num_bins - 1);
                        int psi_idx = clamp(int((psi + M_PI) / 2.f / M_PI * num_bins) + j, 0, num_bins - 1);
                        int phi_idx = clamp(int((phi + M_PI) / 2.f / M_PI * num_bins) + k, 0, num_bins - 1);
                        
                        float phi_delta = phi + M_PI - (M_PI / num_bins + phi_idx * 2.f * M_PI / num_bins);
                        float psi_delta = psi + M_PI - (M_PI / num_bins + psi_idx * 2.f * M_PI / num_bins);
                        float th_delta = r31 + 1.f - (1.f / num_bins + float(th_idx * 2.f) / num_bins);
                        
                        int idx = phi_idx * num_bins * num_bins + th_idx * num_bins + psi_idx;
                        
                        atomicAdd(&zyx_grid[idx], 1.f);
                        atomicAdd(&zyx_delta[idx * 3], phi_delta);
                        atomicAdd(&zyx_delta[idx * 3 + 1], th_delta);
                        atomicAdd(&zyx_delta[idx * 3 + 2], psi_delta);
                    }
                }
            }
            
            // int th_idx = clamp((th + M_PI / 2) / M_PI * num_bins, 0, num_bins - 1);
            // int psi_idx = clamp((psi + M_PI) / 2 / M_PI * num_bins, 0, num_bins - 1);
            // int phi_idx = clamp((phi + M_PI) / 2 / M_PI * num_bins, 0, num_bins - 1);
            
            // atomicAdd(&zyx_grid[phi_idx * num_bins * num_bins + th_idx * num_bins + psi_idx], 1.f);
            
            // another sol.
            // th = M_PI - th;
            // thcos = cos(th);
            // psi = atan2(r32 / thcos, r33 / thcos);
            // phi = atan2(r21 / thcos, r11 / thcos);
            
            // th_idx = clamp((th + M_PI / 2) / M_PI * num_bins, 0, num_bins - 1);
            // psi_idx = clamp((psi + M_PI) / 2 / M_PI * num_bins, 0, num_bins - 1);
            // phi_idx = clamp((phi + M_PI) / 2 / M_PI * num_bins, 0, num_bins - 1);
            
            // atomicAdd(&zyx_grid[phi_idx * num_bins * num_bins + th_idx * num_bins + psi_idx], 1.f);
            
        }
    }
''', 'nocs_rot_voting')


nocs_rot_voting_kernel_direct = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    
    extern "C" __global__
    void nocs_rot_voting(
        const float *points, const float *nocs, const float *center, float *zyx_grid, float *zyx_delta, int *idx2s, int n_samples, int num_bins
    ) {
        const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
        const int err_bound = 3;
        if (idx1 < n_samples) {
            const float3 cent = *((const float3 *)center);
            float3 a = make_float3(nocs[idx1 * 3], nocs[idx1 * 3 + 1], nocs[idx1 * 3 + 2]);
            a /= (length(a) + 1e-9);
            float3 b = make_float3(points[idx1 * 3], points[idx1 * 3 + 1], points[idx1 * 3 + 2]) - cent;
            b /= (length(b) + 1e-9);
            
            float adotb = dot(a, b);
            float3 a2b[3];
            if (adotb > 1 - 1e-7) {
                a2b[0] = make_float3(1, 0, 0);
                a2b[1] = make_float3(0, 1, 0);
                a2b[2] = make_float3(0, 0, 1);
            } else {
                float3 c = cross(a, b);
                float c_norm_sq = dot(c, c);
                float d = (1 - adotb) / (c_norm_sq + 1e-9);
                a2b[0] = make_float3(1, 0, 0) + make_float3(0, -c.z, c.y) + d * make_float3(-c.y * c.y - c.z * c.z, c.x * c.y, c.x * c.z);
                a2b[1] = make_float3(0, 1, 0) + make_float3(c.z, 0, -c.x) + d * make_float3(c.x * c.y, -c.x * c.x - c.z * c.z, c.y * c.z);
                a2b[2] = make_float3(0, 0, 1) + make_float3(-c.y, c.x, 0) + d * make_float3(c.x * c.z, c.y * c.z, -c.x * c.x - c.y * c.y);
            }
            
            float N1 = b.x, N2 = b.y, N3 = b.z;
            for (float theta = 0; theta < 2 * M_PI; theta += 2 * M_PI / num_bins)
            {
                float r11 = a2b[0].x*((cos(theta) - 1)*N2*N2 + (cos(theta) - 1)*N3*N3 + 1) 
                    - a2b[1].x*(N3*sin(theta) + N1*N2*(cos(theta) - 1))
                    + a2b[2].x*(N2*sin(theta) - N1*N3*(cos(theta) - 1));
                float r21 = a2b[1].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N3*N3 + 1) 
                    + a2b[0].x*(N3*sin(theta) - N1*N2*(cos(theta) - 1)) 
                    - a2b[2].x*(N1*sin(theta) + N2*N3*(cos(theta) - 1));
                float r31 = a2b[2].x*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].x*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].x*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                float r32 = a2b[2].y*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].y*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].y*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                float r33 = a2b[2].z*((cos(theta) - 1)*N1*N1 + (cos(theta) - 1)*N2*N2 + 1) 
                    - a2b[0].z*(N2*sin(theta) + N1*N3*(cos(theta) - 1)) 
                    + a2b[1].z*(N1*sin(theta) - N2*N3*(cos(theta) - 1));
                
                float th = -asin(r31);
                float thcos = cos(th);
                float psi = atan2(r32 / thcos, r33 / thcos);
                float phi = atan2(r21 / thcos, r11 / thcos);
                if (isnan(th) || isnan(psi) || isnan(phi)) return;
                
                for (int i = -err_bound; i < err_bound + 1; i++)
                {
                    for (int j = -err_bound; j < err_bound + 1; j++)
                    {
                        for (int k = -err_bound; k < err_bound + 1; k++)
                        {
                            int th_idx = clamp(int((r31 + 1.f) / 2.f * num_bins) + i, 0, num_bins - 1);
                            int psi_idx = clamp(int((psi + M_PI) / 2.f / M_PI * num_bins) + j, 0, num_bins - 1);
                            int phi_idx = clamp(int((phi + M_PI) / 2.f / M_PI * num_bins) + k, 0, num_bins - 1);
                            
                            float phi_delta = phi + M_PI - (M_PI / num_bins + phi_idx * 2.f * M_PI / num_bins);
                            float psi_delta = psi + M_PI - (M_PI / num_bins + psi_idx * 2.f * M_PI / num_bins);
                            float th_delta = r31 + 1.f - (1.f / num_bins + float(th_idx * 2.f) / num_bins);
                            
                            int idx = phi_idx * num_bins * num_bins + th_idx * num_bins + psi_idx;
                            
                            atomicAdd(&zyx_grid[idx], 1.f);
                            atomicAdd(&zyx_delta[idx * 3], phi_delta);
                            atomicAdd(&zyx_delta[idx * 3 + 1], th_delta);
                            atomicAdd(&zyx_delta[idx * 3 + 2], psi_delta);
                        }
                    }
                }
            }
        }
    }
''', 'nocs_rot_voting')


backvote_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void backvote(
        const float *points, const float *outputs, float3 *out_offsets, const int *point_idxs, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z, const float *gt_center, const float tol
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            
            out_offsets[idx] = make_float3(0, 0, 0);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 pred_center = c + offset;
                if (length(pred_center - make_float3(gt_center[0], gt_center[1], gt_center[2])) > tol) continue;
                float3 center_grid = (pred_center - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                    center_grid.x >= grid_x - 1 || center_grid.y >= grid_y - 1 || center_grid.z >= grid_z - 1) {
                    continue;
                }
                out_offsets[idx] = -offset;
                break;
            }
        }
    }
''', 'backvote')

rot_voting_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting(
        const float *points, const float *preds_rot, float3 *outputs_up, const int *point_idxs,
        int n_ppfs, int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= length(ab);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                up = up / (length(up) + 1e-7);
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting')

rot_voting_retrieval_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting_retrieval(
        const float *point_pairs, const float *preds_rot, float3 *outputs_up, 
        int n_ppfs, int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx];
            float3 a = make_float3(point_pairs[idx * 6], point_pairs[idx * 6 + 1], point_pairs[idx * 6 + 2]);
            float3 b = make_float3(point_pairs[idx * 6 + 3], point_pairs[idx * 6 + 4], point_pairs[idx * 6 + 5]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= length(ab);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                up = up / (length(up) + 1e-7);
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting_retrieval')


rot_voting_pair_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting_pair(
        const float *points, 
        const float *preds_rot, 
        float3 *outputs_up, 
        const int *point_idxs,
        int n_ppfs, 
        int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx * 2];
            int a_idx = point_idxs[idx * 4];
            int b_idx = point_idxs[idx * 4 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= length(ab);
            
            float val_rot = preds_rot[idx * 2 + 1];
            int c_idx = point_idxs[idx * 4 + 2];
            int d_idx = point_idxs[idx * 4 + 3];
            float3 c = make_float3(points[c_idx * 3], points[c_idx * 3 + 1], points[c_idx * 3 + 2]);
            float3 d = make_float3(points[d_idx * 3], points[d_idx * 3 + 1], points[d_idx * 3 + 2]);
            float3 cd = c - d;
            if (length(cd) < 1e-7) return;
            cd /= length(cd);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            float best = 1000.f;
            float3 best_up;
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                up = up / (length(up) + 1e-7);
                
                float cand = abs(acos(dot(up, cd)) - val_rot);
                if(cand < best)
                {
                    best = cand;
                    best_up = up;
                }
            }
            outputs_up[idx] = best_up;
        }
    }
''', 'rot_voting_pair')


rot_voting_hand_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting_hand(
        const float *points, const float *preds_rot, const int *preds_aux, float3 *outputs_up, const int *point_idxs,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx];
            int a_idx = point_idxs[idx * 3];
            int b_idx = point_idxs[idx * 3 + 1];
            int c_idx = point_idxs[idx * 3 + 2];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 c = make_float3(points[c_idx * 3], points[c_idx * 3 + 1], points[c_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            float3 abxbc = cross(b - a, c - b);
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + ab;
                // disable
                // if (int(dot(up, abxbc) > 0) ^ preds_aux[idx]) continue;
                up = up / (length(up) + 1e-7);
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting_hand')


findpeak_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void findpeak(
        const float *grids, float *outputs, int width, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < grid_x * grid_y * grid_z) {
            int x = idx / (grid_y * grid_z);
            int yz = idx % (grid_y * grid_z);
            int y = yz / grid_z;
            int z = yz % grid_z;
            float diff_x = grids[idx] - grids[min(grid_x - 1, x + width) * grid_y * grid_z + y * grid_z + z]
                           + grids[idx] - grids[max(0, x - width) * grid_y * grid_z + y * grid_z + z];
            float diff_y = grids[idx] - grids[x * grid_y * grid_z, min(grid_y - 1, y + width) * grid_z + z]
                           + grids[idx] - grids[x * grid_y * grid_z, max(0, y - width) * grid_z + z];
            float diff_z = grids[idx] - grids[x * grid_y * grid_z + y * grid_z + min(grid_z - 1, z + width)]
                           + grids[idx] - grids[x * grid_y * grid_z + y * grid_z + max(0, z - width)];
            outputs[idx] = diff_x + diff_y + diff_z;
        }
    }
''', 'findpeak')


ppf_direct_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting_direct(
        const float *points, const float *outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            int a_idx = point_idxs[idx];
            float3 c = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float prob = probs[idx];
            float3 offset = make_float3(outputs[idx * 3], outputs[idx * 3 + 1], outputs[idx * 3 + 2]);
            float3 center_grid = (c - offset - make_float3(corner[0], corner[1], corner[2])) / res;
            if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                return;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            float lll = w0.x * w0.y * w0.z;
            float llh = w0.x * w0.y * w1.z;
            float lhl = w0.x * w1.y * w0.z;
            float lhh = w0.x * w1.y * w1.z;
            float hll = w1.x * w0.y * w0.z;
            float hlh = w1.x * w0.y * w1.z;
            float hhl = w1.x * w1.y * w0.z;
            float hhh = w1.x * w1.y * w1.z;

            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
        }
    }
''', 'ppf_voting_direct')


backvote_direct_kernel = cp.RawKernel(r'''
    #include "/home/junbo/Codes/A-PIE/BeyondPPF/models/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void backvote_direct(
        const float *points, const float *offsets, float3 *out_offsets, const int *point_idxs, const float *corner, const float res,
        int n_ppfs, int grid_x, int grid_y, int grid_z, const float *gt_center, const float tol
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            int a_idx = point_idxs[idx];
            float3 c = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            
            out_offsets[idx] = make_float3(0, 0, 0);
            float3 offset = make_float3(offsets[idx * 3], offsets[idx * 3 + 1], offsets[idx * 3 + 2]);
            float3 pred_center = c - offset;
            if (length(pred_center - make_float3(gt_center[0], gt_center[1], gt_center[2])) > tol) return;
            float3 center_grid = (pred_center - make_float3(corner[0], corner[1], corner[2])) / res;
            if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                center_grid.x >= grid_x - 1 || center_grid.y >= grid_y - 1 || center_grid.z >= grid_z - 1) {
                return;
            }
            out_offsets[idx] = -offset;
        }
    }
''', 'backvote_direct')
