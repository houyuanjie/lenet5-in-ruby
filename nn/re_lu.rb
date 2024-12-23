module Nn
  class ReLU
    def self.forward(input)
      input.map { |e| e.positive? ? e : 0 }
    end
  end
end
