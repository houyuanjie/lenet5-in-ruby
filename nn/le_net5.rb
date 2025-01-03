require_relative 'conv2d'
require_relative 'max_pool2d'
require_relative 'linear'
require_relative 're_lu2d'
require_relative 're_lu'
require_relative '../matrix/flatten'

module Nn
  class LeNet5
    attr_reader :conv1, :pool1, :conv2, :pool2, :fc1, :fc2, :fc3

    def initialize
      @conv1 = Conv2d.new(in_channels: 1, out_channels: 6, kernel_size: 5, height: 28, width: 28)
      @pool1 = MaxPool2d.new(kernel_size: 2, stride: 2, channels: 6, height: 24, width: 24)

      @conv2 = Conv2d.new(in_channels: 6, out_channels: 16, kernel_size: 5, height: 12, width: 12)
      @pool2 = MaxPool2d.new(kernel_size: 2, stride: 2, channels: 16, height: 8, width: 8)

      @fc1 = Linear.new(in_features: 16 * 4 * 4, out_features: 120)
      @fc2 = Linear.new(in_features: 120, out_features: 84)
      @fc3 = Linear.new(in_features: 84, out_features: 10)
    end

    def forward(input)
      input = [input] if input.is_a?(Matrix)

      last = input.map { |matrix| matrix.map(&:to_f) } # 1 * 28 * 28, channels = 1, height = 28, width = 28

      last = @conv1.forward(last) # 6 * 24 * 24
      last = ReLU2d.forward(last) # 6 * 24 * 24
      last = @pool1.forward(last) # 6 * 12 * 12

      last = @conv2.forward(last) # 16 * 8 * 8
      last = ReLU2d.forward(last) # 16 * 8 * 8
      last = @pool2.forward(last) # 16 * 4 * 4

      last = last.map(&:flatten).flatten # 256

      last = @fc1.forward(last) # 120
      last = ReLU.forward(last) # 120

      last = @fc2.forward(last) # 84
      last = ReLU.forward(last) # 84

      @fc3.forward(last) # 10
    end

    def predict(input)
      output = forward(input)
      output.index(output.max)
    end
  end
end
