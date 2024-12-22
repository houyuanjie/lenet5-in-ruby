require 'matrix'

class Matrix
  def inner_product(other)
    raise 'Matrix dimensions do not match.' unless row_count == other.row_count && column_count == other.column_count

    sum = 0

    row_count.times do |row|
      column_count.times do |column|
        sum += self[row, column] * other[row, column]
      end
    end

    sum
  end

  alias dot inner_product
end
