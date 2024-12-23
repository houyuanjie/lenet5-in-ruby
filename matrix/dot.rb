require 'matrix'

class Matrix
  def frobenius_inner_product(other)
    raise 'Matrix dimensions do not match.' unless row_count == other.row_count && column_count == other.column_count

    sum = 0

    row_count.times do |row_index|
      this_row = rows[row_index]
      other_row = other.rows[row_index]

      column_count.times do |col_index|
        sum += this_row[col_index] * other_row[col_index]
      end
    end

    sum
  end

  alias f_dot frobenius_inner_product
end
