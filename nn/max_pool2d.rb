require 'matrix'
require_relative '../matrix/slice'

module Nn
  class MaxPool2d
    attr_reader :kernel_size, :stride, :channels, :height, :width, :output_height, :output_width

    def initialize(kernel_size:, stride:, channels:, height:, width:)
      @kernel_size = kernel_size
      @stride = stride
      @channels = channels
      @height = height
      @width = width

      @output_height = (height - kernel_size) / stride + 1
      @output_width = (width - kernel_size) / stride + 1
    end

    def forward(input)
      output = Array.new(@channels)

      @channels.times do |chn|
        input_matrix = input[chn]

        pooled_matrix = Matrix.build(@output_height, @output_width) do |row, col|
          row_start = row * @stride
          col_start = col * @stride

          sliced_matrix = input_matrix.slice(
            row_start: row_start, row_length: @kernel_size,
            col_start: col_start, col_length: @kernel_size
          )

          sliced_matrix.max
        end

        output[chn] = pooled_matrix
      end

      output
    end
  end
end
