require 'matrix'

module Mnist
  IMAGE_FILE_MAGIC_NUMBER = 0x0000_0803
  LABEL_FILE_MAGIC_NUMBER = 0x0000_0801

  class LoadImagesResult
    attr_reader :images, :num_images, :num_rows, :num_cols

    def initialize(images, num_images, num_rows, num_cols)
      @images = images
      @num_images = num_images
      @num_rows = num_rows
      @num_cols = num_cols
    end
  end

  class LoadLabelsResult
    attr_reader :labels, :num_labels

    def initialize(labels, num_labels)
      @labels = labels
      @num_labels = num_labels
    end
  end

  def self.load_images(idx_file)
    raise ArgumentError, "File #{idx_file} NOT found." unless File.exist?(idx_file)

    File.open(idx_file) do |file|
      magic_number = file.read(4).unpack1('N')
      raise ArgumentError, "File #{idx_file} is not an image file." unless magic_number == IMAGE_FILE_MAGIC_NUMBER

      num_images = file.read(4).unpack1('N').to_i
      num_rows = file.read(4).unpack1('N').to_i
      num_cols = file.read(4).unpack1('N').to_i

      images = []
      num_images.times do
        image = file.read(num_rows * num_cols).unpack('C*').map(&:to_i)
        rows = image.each_slice(num_cols).to_a
        unless num_rows == rows.size
          raise "Image load failed. Expected #{num_rows} rows, got #{rows.size} rows after split by #{num_cols}."
        end

        matrix = Matrix.rows(rows)
        images << matrix
      end

      raise "Images load failed. Expected #{num_images} images, got #{images.size}." unless num_images == images.size

      return LoadImagesResult.new(images, num_images, num_rows, num_cols)
    end
  end

  def self.load_labels(idx_file)
    raise ArgumentError, "File #{idx_file} NOT found." unless File.exist?(idx_file)

    File.open(idx_file) do |file|
      magic_number = file.read(4).unpack1('N')
      raise ArgumentError, "File #{idx_file} is not a label file." unless magic_number == LABEL_FILE_MAGIC_NUMBER

      num_labels = file.read(4).unpack1('N').to_i
      labels = file.read(num_labels).unpack('C*').map(&:to_i)

      raise "Labels load failed. Expected #{num_labels} labels, got #{labels.size}." unless num_labels == labels.size

      return LoadLabelsResult.new(labels, num_labels)
    end
  end
end
