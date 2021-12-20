/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/attention_layers/TensorParallelUnfusedAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelUnfusedAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                                     const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, d_model],
    //      attention_mask [batch, 1, seqlen, seqlen],
    //      padding_offset [token_num]
    //      relative_attention_bias (optional)
    // If padding_offset.data is nullptr, then not remove padding

    // output_tensors:
    //      attention_out [token_num, d_model]

    UnfusedAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    const size_t size = output_tensors->at(0).shape[0] * output_tensors->at(0).shape[1];
    T* attention_out = (T*)(output_tensors->at(0).data);
    if (tensor_para_size_ > 1) {
        ftNcclAllReduceSum(attention_out, attention_out, size, tensor_para_comm_, UnfusedAttentionLayer<T>::stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelUnfusedAttentionLayer<T>::TensorParallelUnfusedAttentionLayer(size_t max_batch_size,
                                                                            size_t max_seq_len,
                                                                            size_t head_num,
                                                                            size_t size_per_head,
                                                                            size_t d_model,
                                                                            float q_scaling,
                                                                            size_t tensor_para_size,
                                                                            ncclComm_t tensor_para_comm,
                                                                            cudaStream_t stream,
                                                                            cublasMMWrapper* cublas_wrapper,
                                                                            IAllocator* allocator,
                                                                            bool is_free_buffer_after_forward,
                                                                            bool is_sparse):
    UnfusedAttentionLayer<T>(max_batch_size,
                             max_seq_len,
                             head_num / tensor_para_size,
                             size_per_head,
                             d_model,
                             q_scaling,
                             stream,
                             cublas_wrapper,
                             allocator,
                             is_free_buffer_after_forward,
                             is_sparse),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
    FT_CHECK(head_num % tensor_para_size == 0);
}

template<typename T>
TensorParallelUnfusedAttentionLayer<T>::TensorParallelUnfusedAttentionLayer(
    TensorParallelUnfusedAttentionLayer<T> const& attention_layer):
    UnfusedAttentionLayer<T>(attention_layer),
    tensor_para_size_(attention_layer.tensor_para_size_),
    tensor_para_comm_(attention_layer.tensor_para_comm_)
{
}

template class TensorParallelUnfusedAttentionLayer<float>;
template class TensorParallelUnfusedAttentionLayer<half>;

}  // namespace fastertransformer