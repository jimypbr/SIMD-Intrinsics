#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <algorithm>
#include <array>

using std::cout;
using std::endl;
using std::ptrdiff_t;

#define N 71

// pack elements of a 128bit SSE registers left.
// nzeros: number of LHS simd lanes with value 0
static __m128 pack_left_128(__m128 v, int nzeros)
{
    switch (nzeros)
    {
        case 0:
            return v;
        case 1:
            return _mm_permute_ps(v,_MM_SHUFFLE(0,3,2,1));
        case 2:
            return _mm_permute_ps(v,_MM_SHUFFLE(1,0,3,2));
        case 3:
            return _mm_permute_ps(v,_MM_SHUFFLE(2,1,0,3));
        default:
            return v;
    }
}

// bit pattern mask on a SIMD lane for sse masked load/store
static const unsigned int ON = (1<<31);

static const __m128i load_mask_128[] = {_mm_setr_epi32(ON,ON,ON,ON),
                                        _mm_setr_epi32(ON,ON,ON,0),
                                        _mm_setr_epi32(ON,ON,0,0),
                                        _mm_setr_epi32(ON,0,0,0),
                                        _mm_setr_epi32(0,0,0,0),
                                        _mm_setr_epi32(0,0,0,ON),
                                        _mm_setr_epi32(0,0,ON,ON),
                                        _mm_setr_epi32(0,ON,ON,ON)};

static const __m128i store_mask_128[] = {_mm_setr_epi32(ON,ON,ON,ON),
                                        _mm_setr_epi32(ON,ON,ON,0),
                                        _mm_setr_epi32(ON,ON,0,0),
                                        _mm_setr_epi32(ON,0,0,0),
                                        _mm_setr_epi32(0,0,0,0),
                                        _mm_setr_epi32(0,0,0,ON),
                                        _mm_setr_epi32(0,0,ON,ON),
                                        _mm_setr_epi32(0,ON,ON,ON)};

static const __m256i load_mask_256[] = {_mm256_setr_epi32(ON,ON,ON,ON, ON,ON,ON,ON),
                                        _mm256_setr_epi32(ON,ON,ON,ON, ON,ON,ON,0),
                                        _mm256_setr_epi32(ON,ON,ON,ON, ON,ON,0,0),
                                        _mm256_setr_epi32(ON,ON,ON,ON, ON,0,0,0),
                                        _mm256_setr_epi32(ON,ON,ON,ON, 0,0,0,0)};

// print the contents of a 256 vec intrinsic to stdout
inline void print_simd256(__m256 v)
{
    float vec[8];

    _mm256_storeu_ps(&vec[0], v);
    cout << "{";
    for (int i=0; i<8; ++i)
        cout << vec[i] << ", ";
    cout << "}" << endl;
}

// print the contents of a 128 vec intrinsic to stdout
inline void print_simd128(__m128 v)
{
    float vec[4];

    _mm_storeu_ps(&vec[0], v);
    cout << "{";
    for (int i=0; i<4; ++i)
        cout << vec[i] << ", ";
    cout << "}" << endl;
}

// reverse the contents of an avx register (256bit)
inline __m256 reverse_avx(__m256 in)
{
    __m256 perm = _mm256_permute2f128_ps(in, in, 0b00000001);
    __m256 shuf = _mm256_shuffle_ps(perm, perm, 0b00011011);
    return shuf;
}

// perform reversal on an array whose size is <VLEN
void reverse_vectorised_subvector(float *left, float *right)
{
    ptrdiff_t dist = right - left;

    // special case for subvector sized dists.
    if (dist <= 8)
    {
        // inside 128bit vec
        if (dist <= 4)
        {
            unsigned int nzeros = 4 - dist;
            __m128 v = _mm_maskload_ps(left, load_mask_128[nzeros]);
            __m128 rev = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
            __m128 packleft = pack_left_128(rev, nzeros);
            _mm_maskstore_ps(left, store_mask_128[nzeros], packleft);
        }

        // inside 256bit vec
        else
        {
            unsigned int nzeros = 8-dist;
            __m256 v = _mm256_maskload_ps(left, load_mask_256[nzeros]);
            print_simd256(v);
            __m256 rev = reverse_avx(v);
            print_simd256(rev);
            __m128 lo = _mm256_extractf128_ps(rev,0xFFFF0000);
            print_simd128(lo);
            __m128 hi = _mm256_extractf128_ps(rev,0x0000FFFF);
            print_simd128(hi);
            __m128 lo_packleft = pack_left_128(lo, nzeros);
            print_simd128(lo_packleft);
            _mm_storeu_ps(left, lo_packleft);
            _mm_storeu_ps(left+(4-nzeros), hi);
        }
    }
}

// Vectorised array reversal routine.
// range used is [left, right)
// left: start address of the array to be reversed.
// right: end address+1 of the array to be reversed.
void reverse_vectorized(float* left, float* right)
{
    ptrdiff_t dist = right - left;
    if (dist < 0)
        exit(1);

    // execute scalar reverse if dist < ALIGNMENT/2
    if (dist < 16)
    {
        float* left_ = left;
        float* right_ = right - 1;
        while (left_ < right_)
        {
            float tmp = *left_;
            *left_ = *right_;
            *right_ = tmp;
            left_ += 1;
            right_ -=1;
        }
        return;
    }

    // vectorised reverse
    else
    {
        // iterate from both ends and swap(left++,right--)
        // peel the unaligned reversals
        float *lalign = (float *) ( ((uintptr_t) left + 28) - ((uintptr_t)left + 28)%32);
        float* left_ = left;
        float* right_ = right-1;

        while (left_ < lalign)
        {
            float tmp = *left_;
            *left_ = *right_;
            *right_ = tmp;
            left_ += 1;
            right_ -=1;
        }

        // main body with aligned left_ accesses
        while ( (ptrdiff_t) (right_-left_) > 8)
        {
            __m256 bl = _mm256_loadu_ps(left_);
            __m256 br = _mm256_loadu_ps(right_-7);
            __m256 revl = reverse_avx(bl);
            __m256 revr = reverse_avx(br);
            _mm256_storeu_ps(right_-7, revl);
            _mm256_storeu_ps(left_, revr);
            left_ += 8;
            right_ -= 8;
        }

        // remainder loop
        while (left_ < right_)
        {
            float tmp = *left_;
            *left_ = *right_;
            *right_ = tmp;
            left_ += 1;
            right_ -=1;
        }
    }
}

// Rotated the array a length size by offset places to the right.
// offset value is re-adjusted by offload%size
void rotate_vectorized(float *a, int size, int offset)
{
    int offset_ = offset % size;
    reverse_vectorized(&a[0], &a[offset_]);
    reverse_vectorized(&a[offset_], &a[size]);
    reverse_vectorized(&a[0], &a[size]);
}

void reverse_stl(float *a)
{
    std::reverse(&(a[0]), &(a[N]));
}

typedef std::chrono::high_resolution_clock hrclock;

int main()
{
    float a[N] __attribute__((aligned(32)));

    for (int i=0; i<N; ++i)
        a[i] = (float) i;

    cout << "Before:" << endl;
    cout << "{";
    for (int i=0; i<N; ++i)
        cout << a[i] << ", ";
    cout << "}" << endl;

    rotate_vectorized(&a[0], N, 7);
    //reverse_vectorized(&a[20], &a[N]);

    cout << "After:" << endl;
    cout << "{";
    for (int i=0; i<N; ++i)
        cout << a[i] << ", ";
    cout << "}" << endl;

/*    auto t1 = hrclock::now();
    reverse_vectorized(&(a[0]));
    auto t2 = hrclock::now();
    std::chrono::duration<double> diff_vec = t2-t1;

    t1 = hrclock::now();
    reverse_stl(&(a[0]));
    t2 = hrclock::now();
    std::chrono::duration<double> diff_stl = t2-t1;

    cout << "Time for vec reverse : " << diff_vec.count() << endl;
    cout << "Time for stl reverse : " << diff_stl.count() << endl;*/

    return 0;
}