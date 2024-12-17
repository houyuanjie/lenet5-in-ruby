require 'matrix'

module Nn
  class ReLU
    def forward(input)
      unless input.is_a?(Array) || input.is_a?(Vector) || input.is_a?(Matrix)
        raise TypeError, 'Parameter :input must be an Array, Vector, Matrix or Array of Matrix'
      end

      elems = input.map do |e|
        case e
        when Numeric
          e.positive? ? e : 0
        when Matrix
          e.map { |x| x.positive? ? x : 0 }
        else
          raise TypeError, 'Elements of :input must be Numeric or Matrix'
        end
      end

      elems.to_a
    end
  end
end
