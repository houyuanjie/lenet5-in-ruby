require 'matrix'

module Nn
  class Linear
    attr_reader :in_features, :out_features
    attr_accessor :weight, :bias

    def initialize(in_features:, out_features:)
      @in_features = in_features
      @out_features = out_features

      @weight = Matrix.build(out_features, in_features) { rand(-0.1..0.1) }
      @bias = Array.new(out_features) { rand(-0.1..0.1) }
    end

    def forward(input)
      input_v = Vector.elements(input, false)
      bias_v = Vector.elements(@bias, false)

      output_v = @weight * input_v + bias_v
      output_v.to_a
    end
  end
end
