import os
dir_path='./sample_images'
filenames=[x for x in os.listdir(dir_path)]


for filename in filenames :
    fname = filename[:filename.find('.')]
    string = '[' + fname + ']: ' + dir_path + '/' + filename + '  \"'+ fname +'\"'
    print(string)

print()
for filename in filenames :
    fname = filename[:filename.find('.')]
    string = '!['+fname+']['+fname+']'
    print(string)