require 'fileutils'
require 'open-uri'
require 'zlib'

module Mnist
  BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'.freeze

  FILES = {
    train_images: 'train-images-idx3-ubyte.gz',
    train_labels: 'train-labels-idx1-ubyte.gz',
    test_images: 't10k-images-idx3-ubyte.gz',
    test_labels: 't10k-labels-idx1-ubyte.gz'
  }.freeze

  def self.install_to(dest_dir)
    FileUtils.mkdir_p(dest_dir)

    FILES.each_value do |filename|
      dest_file = File.new(File.join(dest_dir, filename.gsub(/\.gz$/, '')))
      next if File.exist?(dest_file)

      puts "Downloading #{filename}..."
      URI.open(URI.join(BASE_URL, filename)) do |remote_file|
        gz_reader = Zlib::GzipReader.new(remote_file)
        FileUtils.copy_stream(gz_reader, dest_file)
      end
    end
  end
end
