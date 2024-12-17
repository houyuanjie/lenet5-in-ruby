require 'matrix'

module Nn
  class Linear
    attr_reader :in_features, :out_features
    attr_accessor :weight, :bias

    def initialize(in_features:, out_features:)
      @in_features = in_features
      @out_features = out_features

      @weight = Matrix.build(out_features, in_features) { rand(-0.1..0.1) }
      @bias = Vector.elements(Array.new(out_features) { rand(-0.1..0.1) })
    end

    def forward(input)
      raise TypeError, 'Parameter :input must be an Array or Vector' unless input.is_a?(Array) || input.is_a?(Vector)

      # NOTE: Linear layer in this case is only used for one dimensional input (Array or Vector).
      #   So we can do a simple implementation.

      input = Vector.elements(input) if input.is_a?(Array)

      input * @weight + @bias
    end
  end
end
