require_relative '../matrix/flatten'

module Nn
  class LeNet5
    def initialize
      @conv1 = Conv2d.new(in_channels: 1, out_channels: 6, kernel_size: 5, height: 28, width: 28)
      @relu1 = ReLU2d.new
      @pool1 = MaxPool2d.new(kernel_size: 2, stride: 2, channels: 6, height: 24, width: 24)

      @conv2 = Conv2d.new(in_channels: 6, out_channels: 16, kernel_size: 5, height: 12, width: 12)
      @relu2 = ReLU2d.new
      @pool2 = MaxPool2d.new(kernel_size: 2, stride: 2, channels: 16, height: 8, width: 8)

      @fc1 = Linear.new(in_features: 16 * 4 * 4, out_features: 120)
      @relu3 = ReLU.new

      @fc2 = Linear.new(in_features: 120, out_features: 84)
      @relu4 = ReLU.new

      @fc3 = Linear.new(in_features: 84, out_features: 10)
      @softmax = Softmax.new
    end

    def forward(input)
      unless input.is_a?(Matrix) || (input.is_a?(Array) && input.all? { |e| e.is_a?(Matrix) })
        raise TypeError, 'Parameter :input must be a Matrix or Array of Matrix'
      end

      input = [input] if input.is_a?(Matrix)

      last = input.map { |matrix| matrix.map(&:to_f) } # 1 * 28 * 28, channels = 1, height = 28, width = 28

      last = @conv1.forward(last) # 6 * 24 * 24
      last = @relu1.forward(last) # 6 * 24 * 24
      last = @pool1.forward(last) # 6 * 12 * 12

      last = @conv2.forward(last) # 16 * 8 * 8
      last = @relu2.forward(last) # 16 * 8 * 8
      last = @pool2.forward(last) # 16 * 4 * 4

      last = last.map(&:flatten).flatten # 256

      last = @fc1.forward(last) # 120
      last = @relu3.forward(last) # 120

      last = @fc2.forward(last) # 84
      last = @relu4.forward(last) # 84

      last = @fc3.forward(last) # 10
      @softmax.forward(last) # 10
    end
  end
end
