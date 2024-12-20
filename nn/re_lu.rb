require 'matrix'

module Nn
  class ReLU
    def forward(input)
      raise TypeError, 'Parameter :input must be an Array' unless input.is_a?(Array)

      input.map { |e| e.positive? ? e : 0 }
    end
  end
end
