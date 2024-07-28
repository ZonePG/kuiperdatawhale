//
// Created by fss on 23-7-22.
//
#include "layer/abstract/layer_factory.hpp"
#include "../source/layer/details/convolution.hpp"
#include <bits/stdint-uintn.h>
#include <gtest/gtest.h>
#include <vector>

using namespace kuiper_infer;

TEST(test_registry, create_layer_convforward) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs(batch_size);

  const uint32_t in_channel = 2;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
    input->data().slice(0) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";

    input->data().slice(1) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";
    inputs.at(i) = input;
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 2;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->data().slice(0) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    kernel->data().slice(1) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    weights.push_back(kernel);
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}

TEST(test_registry, create_layer_convforward_grouped) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs(batch_size);

  const uint32_t in_channel = 2;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
    for (uint32_t j = 0; j < in_channel; ++j) {
      input->data().slice(j) = "1,2,3,4;"
                               "5,6,7,8;"
                               "9,10,11,12;"
                               "13,14,15,16;";
    }
    inputs.at(i) = input;
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 2;
  const uint32_t groups = 2;
  std::vector<sftensor> weights;
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t i = 0; i < kernel_count / groups; ++i) {
      sftensor kernel = std::make_shared<ftensor>(in_channel / groups, kernel_h, kernel_w);
      for (uint32_t j = 0; j < in_channel / groups; ++j) {
        kernel->data().slice(j) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
      }
      weights.push_back(kernel);
    }
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, groups, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}
