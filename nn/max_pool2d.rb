require 'matrix'
require_relative '../matrix/slice'

module Nn
  class MaxPool2d
    attr_reader :kernel_size, :stride, :channels, :height, :width, :out_height, :out_width

    def initialize(kernel_size:, stride:, channels:, height:, width:)
      @kernel_size = kernel_size
      @stride = stride
      @channels = channels
      @height = height
      @width = width

      @out_height = (height - kernel_size) / stride + 1
      @out_width = (width - kernel_size) / stride + 1
    end

    def forward(input)
      raise TypeError, 'Parameter :input must be an Array' unless input.is_a?(Array)
      raise TypeError, 'Elements of :input must be Matrix' unless input.all? { |e| e.is_a?(Matrix) }

      output = Array.new(@channels)

      @channels.times do |chn|
        pooled_matrix = Matrix.build(@out_height, @out_width) do |row, col|
          h_start = row * @stride
          h_end = h_start + @kernel_size
          h_end = [h_end, @height].min

          w_start = col * @stride
          w_end = w_start + @kernel_size
          w_end = [w_end, @width].min

          region = input[chn].slice(row_range: (h_start...h_end), col_range: (w_start...w_end))

          region.max
        end

        output[chn] = pooled_matrix
      end

      output
    end
  end
end
