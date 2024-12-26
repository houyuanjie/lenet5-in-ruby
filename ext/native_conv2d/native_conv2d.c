#include "native_conv2d.h"

VALUE matrix_get_rows(VALUE matrix)
{
  // 直接获取 Matrix 类的内部表示 @rows
  return rb_iv_get(matrix, "@rows");
}

VALUE matrix_make(VALUE rows, int column_count)
{
  // 直接调用 Matrix 类私有的 new 方法，跳过所有检查，并且避免复制
  return rb_funcall(class_matrix, rb_intern("new"), 2, rows, INT2FIX(column_count));
}

VALUE matrix_slice(VALUE matrix, int row_start, int row_length, int col_start, int col_length)
{
  VALUE rows = matrix_get_rows(matrix);
  VALUE sliced_rows = rb_ary_new_capa(row_length);

  for (int row_index = row_start; row_index < row_start + row_length; row_index++)
  {
    VALUE row = rb_ary_entry(rows, row_index);
    VALUE sliced_row = rb_ary_subseq(row, col_start, col_length);
    rb_ary_push(sliced_rows, sliced_row);
  }

  VALUE sliced_matrix = matrix_make(sliced_rows, col_length);

  return sliced_matrix;
}

double matrix_dot(VALUE matrix, VALUE other_matrix, int height, int width)
{
  VALUE rows = matrix_get_rows(matrix);
  VALUE other_rows = matrix_get_rows(other_matrix);

  double sum = 0;

  for (int row_index = 0; row_index < height; row_index++)
  {
    VALUE row = rb_ary_entry(rows, row_index);
    VALUE other_row = rb_ary_entry(other_rows, row_index);

    for (int col_index = 0; col_index < width; col_index++)
    {
      VALUE element = rb_ary_entry(row, col_index);
      VALUE other_element = rb_ary_entry(other_row, col_index);

      sum += rb_num2dbl(element) * rb_num2dbl(other_element);
    }
  }

  return sum;
}

VALUE native_forward(VALUE self,
                     VALUE input_qube, VALUE input_channels, VALUE input_height, VALUE input_width,
                     VALUE output_channels, VALUE output_height, VALUE output_width,
                     VALUE weights, VALUE bias, VALUE kernel_size, VALUE stride)
{
  // 输入通道数
  int num_input_channels = (int)rb_num2int(input_channels);
  // 输入高度
  int num_input_height = (int)rb_num2int(input_height);
  // 输入宽度
  int num_input_width = (int)rb_num2int(input_width);

  // 卷积核尺寸
  int num_kernel_size = (int)rb_num2int(kernel_size);
  // 步长
  int num_stride = (int)rb_num2int(stride);

  // 输出通道数
  int num_output_channels = (int)rb_num2int(output_channels);
  // 输出高度
  int num_output_height = (int)rb_num2int(output_height);
  // 输出宽度
  int num_output_width = (int)rb_num2int(output_width);

  // 输出数据 (3-D, Array[Matrix])
  VALUE output_qube = rb_ary_new_capa(num_output_channels);

  for (int out_chn = 0; out_chn < num_output_channels; out_chn++)
  {
    // 当前通道使用的卷积核组 (3-D, Array[Matrix], length: num_input_channels)
    VALUE kernels = rb_ary_entry(weights, out_chn);
    // 偏置
    VALUE var = rb_ary_entry(bias, out_chn);

    VALUE matrix_rows = rb_ary_new_capa(num_output_height);

    for (int row = 0; row < num_output_height; row++)
    {
      VALUE matrix_row = rb_ary_new_capa(num_output_width);

      for (int col = 0; col < num_output_width; col++)
      {
        int input_matrix_row_start = row * num_stride;
        int input_matrix_col_start = col * num_stride;

        double conv_sum = 0;

        for (int in_chn = 0; in_chn < num_input_channels; in_chn++)
        {
          // 当前通道的输入矩阵 (2-D, Matrix)
          VALUE input_matrix = rb_ary_entry(input_qube, in_chn);
          // 当前通道的卷积核 (2-D, Matrix)
          VALUE kernel = rb_ary_entry(kernels, in_chn);

          VALUE sliced_matrix = matrix_slice(input_matrix,
                                             input_matrix_row_start, num_kernel_size,
                                             input_matrix_col_start, num_kernel_size);

          conv_sum += matrix_dot(sliced_matrix, kernel, num_kernel_size, num_kernel_size);
        }

        VALUE conv_result = rb_float_new(conv_sum + rb_float_value(var));
        rb_ary_push(matrix_row, conv_result);
      }

      rb_ary_push(matrix_rows, matrix_row);
    }

    VALUE matrix = matrix_make(matrix_rows, num_output_width);
    rb_ary_push(output_qube, matrix);
  }

  return output_qube;
}

void Init_native_conv2d()
{
  rb_require("matrix");
  class_matrix = rb_define_class("Matrix", rb_cObject);

  module_nn = rb_define_module("Nn");
  class_conv2d = rb_define_class_under(module_nn, "Conv2d", rb_cObject);
  rb_define_private_method(class_conv2d, "native_forward", native_forward, 11);
}