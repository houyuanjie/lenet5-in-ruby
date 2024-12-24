require 'ruby-prof'
require 'fileutils'

def prof(report_dir, &block)
  FileUtils.mkdir_p(report_dir)

  result = RubyProf::Profile.profile(&block)

  File.open(File.join(report_dir, 'graph.html'), 'w') do |file|
    RubyProf::GraphHtmlPrinter.new(result).print(file)
  end
  File.open(File.join(report_dir, 'stack.html'), 'w') do |file|
    RubyProf::CallStackPrinter.new(result).print(file)
  end
end
