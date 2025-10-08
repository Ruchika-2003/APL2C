#include <stddef.h>
#include <stdint.h>
struct C_CTuple {
    int32_t element_0;
};
struct CNumpyBuffer_float64_1 {
    void* arr;
    double* data;
    size_t length;
    struct C_CTuple shape;
};
double dot_product(struct CNumpyBuffer_float64_1 a, struct CNumpyBuffer_float64_1 b) {
    double c = (double)0.0;
    for (int64_t i = (int64_t)0; i < a.length; i++) {
        size_t linear_idx = 0;
        linear_idx += i;
        size_t linear_idx_2 = 0;
        linear_idx_2 += i;
        c = c + a.data[linear_idx] * b.data[linear_idx_2];
    }
    return c;
}