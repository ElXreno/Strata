#pragma once

#include <cstdint>

namespace strata::kernels {

void lookahead_select(const float* logits, int n_tok, const float* bias, const int32_t* res_layer, float tau, int max_out,
                      int32_t* out, void* stream);

}  // namespace strata::kernels
