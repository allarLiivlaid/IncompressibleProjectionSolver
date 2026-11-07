// hungarian_avx_correct.c
#include <stdio.h>
#include <immintrin.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_free _aligned_free
float* aligned_float_alloc(int n) { return (float*) _aligned_malloc(n * sizeof(float), 32); }
int* aligned_int_alloc(int n) { return (int*) _aligned_malloc(n * sizeof(int), 32); }
#else
#include <stdlib.h>
#define aligned_free free
float* aligned_float_alloc(int n) { return (float*) aligned_alloc(32, ((n*sizeof(float)+31)/32)*32); }
int* aligned_int_alloc(int n) { return (int*) aligned_alloc(32, ((n*sizeof(int)+31)/32)*32); }
#endif

#define BIG 1e20f

// Hungarian algorithm: cost(n x m), assignment[n]
void hungarian_avx(float* cost, int n, int m, int* assignment) {
    float* u = aligned_float_alloc(n);
    float* v = aligned_float_alloc(m);
    int* p = aligned_int_alloc(m + 1);
    int* way = aligned_int_alloc(m + 1);
    float* minv = aligned_float_alloc(m + 1);
    int* used = aligned_int_alloc(m + 1);

    for (int i = 0; i < n; i++) u[i] = 0.0f;
    for (int j = 0; j < m; j++) v[j] = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= m; j++) {
            minv[j] = BIG;
            used[j] = 0;
            way[j] = -1;
        }

        p[0] = i;
        int j0 = 0;

        while (1) {
            used[j0] = 1;
            int i0 = p[j0];
            float u_i0 = u[i0];

            float delta = BIG;
            int j1 = -1;

            // AVX inner loop for minv update
            int j;
            for (j = 1; j + 7 <= m; j += 8) {
                __m256 cost_vec = _mm256_loadu_ps(&cost[i0 * m + j - 1]);
                __m256 u_vec = _mm256_set1_ps(u_i0);
                __m256 v_vec = _mm256_loadu_ps(&v[j]);
                __m256 cur_vec = _mm256_sub_ps(cost_vec, u_vec);
                cur_vec = _mm256_sub_ps(cur_vec, v_vec);

                __m256 minv_vec = _mm256_loadu_ps(&minv[j]);

                // mask unused
                __m256 mask_vec = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i*)&used[j]), _mm256_setzero_si256())
                );
                __m256 cmp = _mm256_cmp_ps(cur_vec, minv_vec, _CMP_LT_OQ);
                __m256 update_mask = _mm256_and_ps(mask_vec, cmp);
                minv_vec = _mm256_blendv_ps(minv_vec, cur_vec, update_mask);
                _mm256_storeu_ps(&minv[j], minv_vec);

                // scalar selection for delta & j1
                float tmp[8];
                _mm256_storeu_ps(tmp, minv_vec);
                for (int k = 0; k < 8; k++) {
                    if (!used[j + k] && tmp[k] < delta) {
                        delta = tmp[k];
                        j1 = j + k;
                    }
                }
            }

            // tail loop
            for (; j <= m; j++) {
                if (!used[j]) {
                    float cur = cost[i0 * m + j - 1] - u_i0 - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // update potentials
            for (int jj = 0; jj <= m; jj++) {
                if (used[jj]) {
                    u[p[jj]] += delta;
                    v[jj] -= delta;
                } else {
                    minv[jj] -= delta;
                }
            }

            j0 = j1;
            if (p[j0] == -1) break;
        }

        // augmenting path
        while (j0 != 0) {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }

    // extract assignment
    for (int i = 0; i < n; i++) assignment[i] = -1;
    for (int j = 1; j <= m; j++) {
        if (p[j] != -1) {
            assignment[p[j]] = j - 1;  // 0-based
        }
    }

    // free memory
    aligned_free(u);
    aligned_free(v);
    aligned_free(p);
    aligned_free(way);
    aligned_free(minv);
    aligned_free(used);
}