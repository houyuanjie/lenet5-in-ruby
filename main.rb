require_relative 'mnist/install'
require_relative 'mnist/load'
require_relative 'nn/le_net5'
require_relative 'model/setup'

Mnist.install_to('./dataset/MNIST/raw')

test_images = Mnist.load_images('./dataset/MNIST/raw/t10k-images-idx3-ubyte')
puts "test_images: num_images=#{test_images.num_images}"
puts "test_images: num_rows=#{test_images.num_rows}"
puts "test_images: num_cols=#{test_images.num_cols}"

test_labels = Mnist.load_labels('./dataset/MNIST/raw/t10k-labels-idx1-ubyte')
puts "test_labels: num_labels=#{test_labels.num_labels}"

model = Nn::LeNet5.new
Nn::LeNet5.setup(model)

def test_model(model, images, labels)
  total_count = labels.num_labels

  correct_count = 0
  images.num_images.times do |i|
    puts "Testing #{i}/#{total_count}" if (i % 200).zero?
    puts "Current correct: #{correct_count}/#{i}" if (i % 200).zero?

    image = images.images[i]
    label = labels.labels[i]

    predicted_label = model.predict(image)
    correct_count += 1 if predicted_label == label
  end

  accuracy = (correct_count.to_f / total_count) * 100
  puts "Test Accuracy: #{accuracy}%"
end

test_model(model, test_images, test_labels)

puts 'Ok'
