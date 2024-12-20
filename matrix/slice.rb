require 'matrix'

class Matrix
  def slice(row_start:, row_length:, col_start:, col_length:)
    rows.slice(row_start, row_length).map { |row| row.slice(col_start, col_length) }
  end
end
