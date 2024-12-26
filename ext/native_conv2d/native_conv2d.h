#ifndef NATIVE_CONV2D_H
#define NATIVE_CONV2D_H

#include <ruby.h>

// class Matrix
VALUE class_matrix;

// module Nn
VALUE module_nn;
// class Conv2d
VALUE class_conv2d;

// Conv2d#native_forward
VALUE native_forward(VALUE module,
                     VALUE input, VALUE input_channels, VALUE input_height, VALUE input_width,
                     VALUE output_channels, VALUE output_height, VALUE output_width,
                     VALUE weights, VALUE bias, VALUE kernel_size, VALUE stride);

void Init_native_conv2d();

#endif
