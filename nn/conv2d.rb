require 'matrix'
require_relative '../matrix/slice'
require_relative '../matrix/dot'

module Nn
  class Conv2d
    attr_reader :in_channels, :out_channels, :kernel_size, :height, :width, :stride, :padding,
                :output_height, :output_width
    attr_accessor :weights, :bias

    def initialize(in_channels:, out_channels:, kernel_size:, height:, width:, stride: 1, padding: 0)
      @in_channels = in_channels
      @out_channels = out_channels
      @kernel_size = kernel_size
      @height = height
      @width = width
      @stride = stride
      @padding = padding

      @output_height = ((height - kernel_size + 2 * padding) / stride) + 1
      @output_width = ((width - kernel_size + 2 * padding) / stride) + 1

      @weights = Array.new(out_channels) { Array.new(in_channels) { Matrix.zero(kernel_size, kernel_size) } }
      @bias = Array.new(out_channels) { 0 }
    end

    def forward(input)
      input = padded_input(input)

      Array.new(@out_channels) do |out_chn|
        kernels = @weights[out_chn]
        var = @bias[out_chn]

        Matrix.build(@output_height, @output_width) do |row, col|
          row_start = row * stride
          col_start = col * stride

          conv_sum = 0

          @in_channels.times do |in_chn|
            input_matrix = input[in_chn]
            kernel = kernels[in_chn]

            sliced_matrix = input_matrix.slice(
              row_start: row_start, row_length: @kernel_size,
              col_start: col_start, col_length: @kernel_size
            )

            conv_sum += sliced_matrix.f_dot(kernel)
          end

          conv_sum + var
        end
      end
    end

    private

    def padded_input(input)
      return input if @padding.zero?

      padded_height = @height + 2 * @padding
      padded_width = @width + 2 * @padding

      Array.new(@in_channels) do |chn|
        input_matrix = input[chn]

        Matrix.build(padded_height, padded_width) do |row, col|
          if row < @padding || padded_height - @padding <= row || col < @padding || padded_width - @padding <= col
            0
          else
            input_matrix[row - @padding, col - @padding]
          end
        end
      end
    end
  end
end
