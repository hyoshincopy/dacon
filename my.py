from sklearn.utils import shuffle
import numpy as np
data_base_path = 'data'

with open('train.csv', 'r') as csv_file:
    data = csv_file.read()

all_captions = []
all_img_name_vector = []
for line in data.split('\n')[1:-1]:
    image_id, smiles = line.split(',')
    caption = '<' + smiles + '>'
    all_img_name_vector.append(image_id)
    all_captions.append(caption)

#! all_img_name_vector : train_0이 들어있는 리스트
#! all_captions : 분자식이 들어있는 리스트
# print(all_img_name_vector,all_captions,end='\n')
#! shuffle 함수를 통해 데이터를 섞어준다. random_state 값은 난수의 시드값이라고 생각하면 된다.
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=20)

num_examples = 5 # 학습에 사용할 데이터 수, Baseline에서는 제공된 데이터 모두 사용하였습니다.

#! train_captions 와 img_name_vector 를 그냥 리스트가 아닌 numpy 배열로 바꿈 
train_captions = np.array(train_captions[:num_examples])
img_name_vector = np.array(img_name_vector[:num_examples])

#! 데이터를 두 칸씩 띄어서 저장함(아마도 속도때문?) ex) 0 1 2 3 4 가 아닌 0 2 4 만 저장
img_name_vector = img_name_vector[::2]
train_captions = train_captions[::2]

def calc_max_length(tensor):
    return max(len(t) for t in tensor)