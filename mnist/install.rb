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
      dest_file = File.join(dest_dir, filename)

      filename_unzip = filename.gsub(/\.gz$/, '')
      dest_file_unzip = File.join(dest_dir, filename_unzip)

      case [File.exist?(dest_file), File.exist?(dest_file_unzip)]
      in [true, true]
        puts "File #{filename} already downloaded and unzipped."
      in [true, false]
        puts "File #{filename} already downloaded, but not unzipped. Unzipping now..."
        Zlib::GzipReader.open(dest_file) { |gz| FileUtils.copy_stream(gz, dest_file_unzip) }
      in [false, unzipped]
        FileUtils.rm(dest_file_unzip) if unzipped

        puts "Downloading #{filename}..."
        URI.open(URI.join(BASE_URL, filename)) { |remote_file| FileUtils.copy_stream(remote_file, dest_file) }

        puts "Unzipping #{filename}..."
        Zlib::GzipReader.open(dest_file) { |gz| FileUtils.copy_stream(gz, dest_file_unzip) }
      else
        raise 'Unexpected state.'
      end
    end
  end
end
