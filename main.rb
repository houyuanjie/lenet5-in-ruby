require_relative 'mnist/install'
require_relative 'mnist/load'

Mnist.install_to('.tmp')

train_images = Mnist.load_images('.tmp/train-images-idx3-ubyte')
puts "train_images: num_images=#{train_images.num_images}"
puts "train_images: num_rows=#{train_images.num_rows}"
puts "train_images: num_cols=#{train_images.num_cols}"

train_labels = Mnist.load_labels('.tmp/train-labels-idx1-ubyte')
puts "train_labels: num_labels=#{train_labels.num_labels}"

test_images = Mnist.load_images('.tmp/t10k-images-idx3-ubyte')
puts "test_images: num_images=#{test_images.num_images}"
puts "test_images: num_rows=#{test_images.num_rows}"
puts "test_images: num_cols=#{test_images.num_cols}"

test_labels = Mnist.load_labels('.tmp/t10k-labels-idx1-ubyte')
puts "test_labels: num_labels=#{test_labels.num_labels}"

puts 'Ok'
