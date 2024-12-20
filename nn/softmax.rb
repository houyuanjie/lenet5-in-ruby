module Nn
  class Softmax
    def forward(input)
      exp_values = input.map { |e| Math.exp(e.to_f).to_f }
      exp_sum = exp_values.sum
      exp_values.map { |e| (e.to_f / exp_sum).to_f }
    end
  end
end
