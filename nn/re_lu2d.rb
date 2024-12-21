require 'matrix'

module Nn
  class ReLU2d
    def self.forward(input)
      input.map { |matrix| matrix.map { |e| e.positive? ? e : 0 } }
    end
  end
end
