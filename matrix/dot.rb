require 'matrix'

class Matrix
  def frobenius_inner_product(other)
    this_rows = rows
    other_rows = other.rows

    num_row_count = this_rows.size
    num_col_count = column_count

    sum = 0

    num_row_count.times do |r|
      this_row = this_rows[r]
      other_row = other_rows[r]

      num_col_count.times do |c|
        sum += this_row[c] * other_row[c]
      end
    end

    sum
  end

  alias f_dot frobenius_inner_product
end
