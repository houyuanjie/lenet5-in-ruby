require 'matrix'

module Nn
  class ReLU
    def forward(input)
      raise TypeError, 'Parameter :input must be an Array' unless input.is_a?(Array)

      input.map do |e|
        case e
        when Numeric
          e.positive? ? e : 0
        when Matrix
          e.map { |x| x.positive? ? x : 0 }
        else
          raise TypeError, 'Elements of :input must be Numeric or Matrix'
        end
      end
    end
  end
end
