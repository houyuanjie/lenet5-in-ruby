require 'fileutils'

# 安装 MNIST 数据集

require_relative 'mnist/install'

mnist_dest_dir = 'dataset/MNIST/raw'

puts 'Downloading MNIST dataset...'
Mnist.install_to(mnist_dest_dir)

# 训练

pyvenv = '.pyvenv'

unless Dir.exist?(pyvenv)
  puts 'Installing Python venv...'
  system('python --version')
  system('python -m venv .pyvenv')
end

is_windows = RUBY_PLATFORM =~ /windows|mswin|mingw/

pip = "#{pyvenv}/Scripts/pip"
pip += '.exe' if is_windows

puts 'Installing Python dependencies...'
system("#{pip} install -r requirements.txt")

setup_rb = 'model/setup.rb'
setup_rb_bak = "#{setup_rb}.bak"

if File.exist?(setup_rb)
  FileUtils.rm(setup_rb_bak) if File.exist?(setup_rb_bak)
  FileUtils.mv(setup_rb, setup_rb_bak)
end

python = "#{pyvenv}/Scripts/python"
python += '.exe' if is_windows

puts 'Training model...'
system("#{python} train.py")

raise "Not found #{setup_rb} after training" unless File.exist?(setup_rb)

# 加载测试集数据

require_relative 'mnist/load'

test_images = Mnist.load_images("#{mnist_dest_dir}/t10k-images-idx3-ubyte")
puts "test_images: num_images=#{test_images.num_images}"
puts "test_images: num_rows=#{test_images.num_rows}"
puts "test_images: num_cols=#{test_images.num_cols}"

test_labels = Mnist.load_labels("#{mnist_dest_dir}/t10k-labels-idx1-ubyte")
puts "test_labels: num_labels=#{test_labels.num_labels}"

# 测试模型

require_relative 'nn/le_net5'
require_relative 'model/setup'

model = Nn::LeNet5.new
Nn::LeNet5.setup(model)

# 测量执行情况

require_relative 'misc/prof'

prof('report') do
  model.predict(test_images.images[0])
end

# 测试准确度

total_count = test_labels.num_labels

correct_count = 0
test_images.num_images.times do |i|
  if (i % 200).zero?
    puts "Testing #{i}/#{total_count}"
    puts "Current correct: #{correct_count}/#{i}"
  end

  image = test_images.images[i]
  label = test_labels.labels[i]

  predicted_label = model.predict(image)
  correct_count += 1 if predicted_label == label
end

puts '-' * 80
puts "Total correct: #{correct_count}/#{total_count}"

accuracy = (correct_count.to_f / total_count) * 100
puts "Test Accuracy: #{accuracy}%"

puts 'Ok'
