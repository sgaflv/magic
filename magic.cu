#include <stdio.h>
#include <stdlib.h>
#include <time.h>       /* time */
#include <iostream>
#include <chrono>
#include <unistd.h>


typedef unsigned long long int u64;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
typedef unsigned long long int BBoard;

#define INDEX_BITS 11
#define IDX 61
#define INITIAL_COUNTER 0x0007ffc000000000ULL

//#define INDEX_BITS 10
//#define IDX 62
//#define INITIAL_COUNTER 0x0003ffef00000000ULL

//#define INDEX_BITS 10
//#define IDX 55
//#define INITIAL_COUNTER 0x510FFFF5F0000000ULL


#define ELEM_COUNT (1ULL << INDEX_BITS)

__device__ void add_bit(BBoard &board, int x, int y) {
    board |= 1ull << (x + y * 8);
}

__device__ bool has_bit(BBoard &board, int x, int y) {
    return (board & (1ull << (x + y * 8))) > 0;
}

__device__ bool d_has_bit(BBoard &board, int x, int y) {
    return (board & (1ull << (x + y * 8))) > 0;
}

__device__ void bb_print(BBoard board) {

        printf("\n");

    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            if (d_has_bit(board, x, 7-y)) {
                printf("*");
            } else {
                printf(".");
            }
        }

        printf("\n");
    }
}


__device__ BBoard get_rook_premask(int idx) {

    BBoard result = 0;

    int x = (idx % 8);
    int y = (idx / 8);

    for (int i=-8; i<8; i++) {
        if (i == 0) continue;

        if (y + i > 0 && y + i < 7) {
            add_bit(result, x, y + i);
        }

        if (x + i > 0 && x + i < 7) {
            add_bit(result, x + i, y);
        } 
    }

    return result;
}

__device__ BBoard get_rook_attack_bits(BBoard indexed_mask, int x, int y) {

    bool slide_n = true;
    bool slide_s = true;
    bool slide_e = true;
    bool slide_w = true;

    BBoard result = 0;

    for (int i = 1; i < 8; i++) {

        if (slide_e && x + i < 8) {

            add_bit(result, x + i, y);

            if (has_bit(indexed_mask, x + i, y)) {
                slide_e = false;
            }
        }

        if (slide_w && x - i >= 0) {

            add_bit(result, x - i, y);

            if (has_bit(indexed_mask, x - i, y)) {
                slide_w = false;
            }
        }

        if (slide_n && y + 1 < 8) {

            add_bit(result, x, y + i);

            if (has_bit(indexed_mask, x, y + i)) {
                slide_n = false;
            }
        }

        if (slide_s && y - 1 >= 0) {

            add_bit(result, x, y - i);

            if (has_bit(indexed_mask, x, y - i)) {
                slide_s = false;
            }
        }

    }

    return result;
}

__device__ BBoard get_indexed_mask(BBoard pre_mask, int mask_number) {

    BBoard result = 0ULL;

    while (pre_mask > 0) {
        BBoard last_bit = pre_mask & -pre_mask;

        bool is_present = mask_number % 2 == 1;

        if (is_present) {
            result |= last_bit;
        }
        mask_number >>= 1;
        pre_mask ^= last_bit;
    }

    return result;
};



__global__ void init_magic_search(BBoard *d_indexed_mask, BBoard *d_attack_bits, int idx, int index_bits) {

    BBoard pre_mask = get_rook_premask(idx);

    int max_test_index = 1 << __popcll(pre_mask);

    int step = 0;
    int x = idx % 8;
    int y = idx / 8;

    while (step < max_test_index) {

        BBoard indexed_mask_e = get_indexed_mask(pre_mask, step);
        d_indexed_mask[step] = indexed_mask_e;

        BBoard attack_bits_e = get_rook_attack_bits(indexed_mask_e, x, y);
        d_attack_bits[step] = attack_bits_e;

        step += 1;
    }
}


__device__ bool check_rook_magic(BBoard *d_indexed_mask, BBoard *d_attack_bits, u64 magic, int max_test_index, int index_bits) {
    
    int index_size = 1ull << index_bits;

    if (index_size > ELEM_COUNT) {
        return false;
    }
    
    BBoard solution[ELEM_COUNT] {};

    int step = 0;
    int shift_bits = 64 - index_bits;

    while (step < max_test_index) {

        BBoard indexed_mask_e = d_indexed_mask[step];
        BBoard rook_attack_mask_e = d_attack_bits[step];

        int index = (indexed_mask_e * magic) >> shift_bits;

        BBoard cur_value = solution[index];

        if (cur_value > 0) {

            if (cur_value == rook_attack_mask_e) {
                step += 1;
                continue;
            } else {

                return false;
            }
        }

        solution[index] = rook_attack_mask_e;

        step += 1;
    }

    return true;
}


__global__ void find_magic(BBoard *d_indexed_mask, BBoard *d_attack_bits, int idx, int index_bits, u64 initial_counter, u64 *result) {

    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    BBoard pre_mask = get_rook_premask(idx);

    int max_test_index = 1ull << __popcll(pre_mask);

    if (*result != 0) {
        
        return;
    }

//////////////////////////////
    u64 magic_candidate = initial_counter + tid;

    bool is_good_magic = check_rook_magic(d_indexed_mask, d_attack_bits, magic_candidate, max_test_index, index_bits);

    if (is_good_magic) {
        printf("blockIdx = %d blockDim = %d threadIdx = %d tid * 2 = %d magic=0x%llx\n", blockIdx.x, blockDim.x, threadIdx.x, tid, magic_candidate);

        *result = magic_candidate;
    }


}


u64 getRandom()
{
    return (((u64)(unsigned int)rand() << 32) + (u64)(unsigned int)rand());
}

int main()
{
    /////////////////////////////////////////
    int idx = IDX;
    int index_bits = INDEX_BITS;
    u64 initial_counter = INITIAL_COUNTER;
    
    /////////////////////////////////////////

    BBoard *d_attack_bits;
    BBoard *d_indexed_mask;

    cudaMalloc(&d_attack_bits, 4096 * sizeof(BBoard));
    cudaMalloc(&d_indexed_mask, 4096 * sizeof(BBoard));

    init_magic_search<<<1, 1>>>(d_indexed_mask, d_attack_bits, idx, index_bits);

    printf("Start magic search\n");

    u64 result = 1;
    u64* d_result;

    int size = sizeof(u64);

    cudaMalloc(&d_result, size);

    srand (time(NULL));
    u64 total_tests = 0;

    u64 initial_check_value = getRandom();

    initial_check_value = initial_counter;
    while (1) {

        u64 block_count = 10000;
        u64 block_size = 512;
        u64 sample_count = block_count * block_size;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        find_magic<<<block_count, block_size>>>(d_indexed_mask, d_attack_bits, idx, index_bits, initial_check_value, d_result);

        cudaMemcpy(&result, d_result, size, cudaMemcpyDeviceToHost);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        u64 speed = 1000000 * sample_count / std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        total_tests += sample_count;

        printf("Testing: 0x%llx - 0x%llx, ", initial_check_value, initial_check_value + sample_count * 2);

        printf("%lld samples per second, %lld total tests\n", speed, total_tests);

        if (result != 0) {
            break;
        }


        initial_check_value += sample_count * 2;
    }

    cudaFree(d_attack_bits);
    cudaFree(d_indexed_mask);
    cudaFree(d_result);

    printf("Magic search finished!\n");
    printf("Result: 0x%llx!\n", result);

}
