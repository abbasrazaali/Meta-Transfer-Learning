# loading datas.json
import json
import os
from PIL import Image

source_path = '/Users/aali/Data/Professional/ADFG/Projects/Vision/MetaVision/corpus/chestxray/'
target_path = '/Users/aali/Data/Professional/ADFG/Projects/Vision/MetaVision/data/chestxray/'

dic = {}
with open('datas.json') as data_file:
    data = json.load(data_file)
    lst = []
    n=0
    print(len(data))
    for i in data:
        # print i
        dic[i] = "/".join(data[i]['items'])

# Labeling the data and forming a distionary
c = 0
st = ""
new_dict = {}
check = ["normal", "opacity", "cardiomegaly", "calcinosis", "lung/hypoinflation", "calcified granuloma",
         "thoracic vertebrae/degenerative", "lung/hyperdistention", "spine/degenerative ", "catheters, indwelling",
         "granulomatous disease", "nodule", "surgical instruments", "scoliosis", "spondylosis"]

for j in dic:
    if(dic[j].lower() in 'normal'):
        new_dict[j] = "normal"
    else:
        new_dict[j] = "nodule"

# training and test folders
# main_dir = '../'
# gConfig = getConfig(main_dir + 'config/metavision.ini')  # get configuration
# testset_proportion = gConfig['testset_proportion']

count = 0
total_files = len(os.listdir(source_path)) - 1
test_file = int(total_files / (total_files * 20 / 100))

for key, value in new_dict.items():
    if not os.path.exists(target_path + '/test/' + value):
        os.makedirs(target_path + '/test/' + value)

    if not os.path.exists(target_path + '/train/' + value):
        os.makedirs(target_path + '/train/' + value)

    if(count % test_file != 0):
        try:
            im = Image.open(os.path.join(source_path, key + '.png'))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(target_path, 'train', value, key + '.jpg'))
            # os.system('cp ' + os.path.join(source_path, key + '.png') + ' ' + os.path.join(target_path, 'test', value, key + '.png'))
        except Exception as e:
            print(str(e))
    else:
        # os.system('cp ' + os.path.join(source_path, key + '.png') + ' ' + os.path.join(target_path, 'train', value, key + '.png'))
        try:
            im = Image.open(os.path.join(source_path, key + '.png'))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(target_path, 'test', value, key + '.jpg'))
        except Exception as e:
            print(str(e))
    count += 1
