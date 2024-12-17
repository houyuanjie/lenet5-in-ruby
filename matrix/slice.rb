require 'matrix'

class Matrix
  def slice(row_range:, col_range:)
    raise TypeError, 'Parameter :row_range must be a Range' unless row_range.is_a?(Range)
    raise TypeError, 'Parameter :col_range must be a Range' unless col_range.is_a?(Range)

    rows = row_range.map { |row_index| row(row_index).to_a.slice(col_range) }
    Matrix.rows(rows)
  end
end
