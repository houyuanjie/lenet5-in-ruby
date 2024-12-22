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
      # TODO: Find out why Matrix.column_vector is not working here.

      input_vector = Matrix.row_vector(input)
      weight_transpose = @weight.transpose
      bias_vector = Matrix.row_vector(@bias)

      output_vector = input_vector * weight_transpose + bias_vector
      raise 'Linear layer output is not a vector.' unless output_vector.row_count == 1

      output_vector.row(0).to_a
    end
  end
end
