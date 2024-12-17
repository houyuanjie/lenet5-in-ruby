module Nn
  class Softmax
    def forward(input)
      raise TypeError, 'Parameter :input must be an Array or Vector' unless input.is_a?(Array) || input.is_a?(Vector)
      raise TypeError, 'Elements of :input must be Numeric' unless input.all? { |e| e.is_a?(Numeric) }

      exp_values = input.map { |e| Math.exp(e) }
      exp_sum = exp_values.sum
      exp_values.map { |e| e / exp_sum }.to_a
    end
  end
end
