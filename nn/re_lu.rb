require 'matrix'

module Nn
  class ReLU
    def forward(input)
      input.map { |e| e.positive? ? e : 0 }
    end
  end
end
