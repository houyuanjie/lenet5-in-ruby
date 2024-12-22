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
      # 矩阵乘法要求输入向量的列数 (input.length) 等于权重矩阵转置后的行数 (@in_features)
      # 因此，我们使用 Matrix.row_vector 将输入转换为 1 行矩阵
      # 得到的结果同样是一个 1 行矩阵，取得其第 1 行，即结果向量

      input_vector = Matrix.row_vector(input)
      weight_transpose = @weight.transpose
      bias_vector = Matrix.row_vector(@bias)

      output_vector = input_vector * weight_transpose + bias_vector
      output_vector.row(0).to_a
    end
  end
end
