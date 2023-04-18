/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/whisper/WhisperTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <vector>

namespace ft = fastertransformer;

template<typename T>
WhisperTritonModelInstance<T>::WhisperTritonModelInstance(std::unique_ptr<ft::WhisperEncoder<T>>        whisper_encoder,
                                                    std::unique_ptr<ft::WhisperDecoding<T>>       whisper_decoding,
                                                    std::shared_ptr<ft::WhisperEncoderWeight<T>>  whisper_encoder_weight,
                                                    std::shared_ptr<ft::WhisperDecodingWeight<T>> whisper_decoding_weight,
                                                    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map,
                                                    std::unique_ptr<std::mutex>          cublas_wrapper_mutex,
                                                    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                    std::unique_ptr<cudaDeviceProp>      cuda_device_prop_ptr):
    whisper_encoder_(std::move(whisper_encoder)),
    whisper_decoding_(std::move(whisper_decoding)),
    whisper_encoder_weight_(whisper_encoder_weight),
    whisper_decoding_weight_(whisper_decoding_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
ft::TensorMap WhisperTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    // copy from Triton sensor to FT tensor
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("sequence_length"), d_input_lengths_, &allocator_);

    // wrap as Tensor class
    ft::TensorMap ft_input_tensors(
        {{"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
         {"sequence_length", as_GPU_tensor(input_tensors->at("sequence_length"), d_input_lengths_)}});

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
WhisperTritonModelInstance<T>::convert_outputs(ft::TensorMap& output_tensors)
{
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
WhisperTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t mem_max_seq_len    = input_tensors->at("input_ids").shape[1];
    const size_t max_output_len     = *((uint*)input_tensors->at("max_output_len").data);
    const size_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;

    allocateBuffer(request_batch_size, beam_width, max_output_len, mem_max_seq_len);

    ft::TensorMap encoder_input_tensors(convert_inputs(input_tensors));

    ft::TensorMap encoder_output_tensors(
        {{"output_hidden_state",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::getTensorType<T>(),
                     std::vector<size_t>{request_batch_size, mem_max_seq_len, whisper_encoder_->getDModel()},
                     d_encoder_outputs_}}});

    ft::TensorMap decoding_input_tensors({{"encoder_output", encoder_output_tensors.at("output_hidden_state")},
                                          {"encoder_sequence_length", encoder_input_tensors.at("sequence_length")}});

    if (input_tensors->find("top_p_decay") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_decay"), d_top_p_decay_, &allocator_);
        decoding_input_tensors.insert({"top_p_decay", as_GPU_tensor(input_tensors->at("top_p_decay"), d_top_p_decay_)});
    }
    if (input_tensors->find("top_p_min") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_min"), d_top_p_min_, &allocator_);
        decoding_input_tensors.insert({"top_p_min", as_GPU_tensor(input_tensors->at("top_p_min"), d_top_p_min_)});
    }
    if (input_tensors->find("top_p_reset_ids") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_, &allocator_);
        decoding_input_tensors.insert(
            {"top_p_reset_ids", as_GPU_tensor(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_)});
    }

    std::set<std::string> keys_on_gpu = {"input_ids",
                                         "sequence_length",
                                         "bad_words_list",
                                         "stop_words_list",
                                         "top_p_decay",
                                         "top_p_min",
                                         "top_p_reset_ids"};
    for (auto& t : *input_tensors) {
        if (keys_on_gpu.count(t.first) == 0) {
            decoding_input_tensors.insert({t.first, t.second.convertTritonTensorToFt()});
        }
    }

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_, &allocator_);
        decoding_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_, &allocator_);
        decoding_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    ft::TensorMap decoding_output_tensors(
        {{"output_ids",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::TYPE_INT32,
                     std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                     d_output_ids_}},
         {"sequence_length",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::TYPE_INT32,
                     std::vector<size_t>{request_batch_size, beam_width},
                     d_sequence_lengths_}}});
    if (input_tensors->count("is_return_log_probs") > 0
        && input_tensors->at("is_return_log_probs").convertTritonTensorToFt().getVal<bool>()) {
        decoding_output_tensors.insert({"output_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                                                   d_output_log_probs_}});
        decoding_output_tensors.insert({"cum_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width},
                                                   d_cum_log_probs_}});
    }

    try {
        whisper_encoder_->forward(&encoder_output_tensors, &encoder_input_tensors, whisper_encoder_weight_.get());
        whisper_decoding_->forward(&decoding_output_tensors, &decoding_input_tensors, whisper_decoding_weight_.get());
    }
    catch (...) {
        h_exception_ = std::current_exception();
        decoding_output_tensors.insert(
            {"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(decoding_output_tensors);
}

template<typename T>
WhisperTritonModelInstance<T>::~WhisperTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void WhisperTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                const size_t beam_width,
                                                const size_t max_output_len,
                                                const size_t mem_max_seq_len)
{
    d_output_ids_      = (int*)(allocator_->reMalloc(
        d_output_ids_, sizeof(int) * request_batch_size * beam_width * max_output_len, false));
    d_encoder_outputs_ = (T*)(allocator_->reMalloc(
        d_encoder_outputs_, sizeof(T) * request_batch_size * mem_max_seq_len * whisper_encoder_->getDModel(), false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * max_output_len, false));
    d_cum_log_probs_    = (float*)(allocator_->reMalloc(
        d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width * max_output_len, false));
    d_within_range_     = (bool*)(allocator_->reMalloc(d_within_range_, sizeof(bool)));
}

template<typename T>
void WhisperTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_encoder_outputs_));
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
    allocator_->free((void**)(&d_within_range_));
}

template struct WhisperTritonModelInstance<float>;
template struct WhisperTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct WhisperTritonModelInstance<__nv_bfloat16>;
#endif