require 'matrix'

module Nn
  class Linear
    attr_reader :in_features, :out_features
    attr_accessor :weight, :bias

    def initialize(in_features:, out_features:)
      @in_features = in_features
      @out_features = out_features

      @weight = Matrix.zero(out_features, in_features)
      @bias = Array.new(out_features) { 0 }
    end

    def forward(input)
      input_v = Vector.elements(input, false)
      bias_v = Vector.elements(@bias, false)

      output_v = @weight * input_v + bias_v
      output_v.to_a
    end
  end
end
