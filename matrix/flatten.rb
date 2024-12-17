require 'matrix'

class Matrix
  def flatten
    flat_map { |e| e }
  end
end
