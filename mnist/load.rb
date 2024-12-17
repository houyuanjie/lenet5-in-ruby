module Mnist
  IMAGE_FILE_MAGIC_NUMBER = 0x0000_0803
  LABEL_FILE_MAGIC_NUMBER = 0x0000_0801

  class LoadImagesResult < Data.define(:images, :num_images, :num_rows, :num_cols)
  end

  class LoadLabelsResult < Data.define(:labels, :num_labels)
  end

  def self.load_images(idx_file)
    raise ArgumentError, "File #{idx_file} NOT found." unless File.exist?(idx_file)

    File.open(idx_file) do |file|
      magic_number = file.read(4).unpack1('N')
      raise ArgumentError, "File #{idx_file} is not an image file." unless magic_number == IMAGE_FILE_MAGIC_NUMBER

      num_images = file.read(4).unpack1('N')
      num_rows = file.read(4).unpack1('N')
      num_cols = file.read(4).unpack1('N')
      image_size = num_rows * num_cols

      images = []
      num_images.times do
        image = file.read(image_size).unpack('C*')
        images << image
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

      num_labels = file.read(4).unpack1('N')
      labels = file.read(num_labels).unpack('C*')

      raise "Labels load failed. Expected #{num_labels} labels, got #{labels.size}." unless num_labels == labels.size

      return LoadLabelsResult.new(labels, num_labels)
    end
  end
end
