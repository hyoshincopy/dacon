from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import rdkit
from sklearn.model_selection import train_test_split
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
# img_name_vector = img_name_vector[::2]
# train_captions = train_captions[::2]

#! 학습에 사용하는 SMILES의 길이는 40이하로 샘플링 된 상태 (근데 왜 하는지는 모르겠음)
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
max_length = calc_max_length(train_captions)

#! tokenizer 를 통해 원소기호와 숫자를 mapping 한다. 
#! ex) (<)-(9) (C)-(1)
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, char_level=True)
tokenizer.fit_on_texts(train_captions)
top_k = len(tokenizer.word_index)
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


#! mol_train 에는 분자식이 들어있고 (학습 데이터)
#! mol_val 에는 분자식이 들어있고 (유효성 테스트 데이터)
#! cap_train 에는 tokenizer 된 숫자로 이루어진 리스트가 들어있고(학습데이터)
#! cap_val 에는 tokenizer 된 숫자로 이루어진 리스트가 들어있고(유효성 테스트 데이터)
mol_train, mol_val, cap_train, cap_val = train_test_split(train_captions, cap_vector, test_size=0.2, random_state=42)
len(mol_train), len(cap_train), len(mol_val), len(cap_val)

#! 하이퍼 파라미터와 학습에 필요한 변수들 설정 
BATCH_SIZE = 128
BUFFER_SIZE = 100
embedding_dim = 256 # Original: 512
units = 1024
vocab_size = top_k + 1
num_steps = len(mol_train) # BATCH_SIZE
features_shape = 2048
attention_features_shape = 64
EPOCHS = 50
learning_rate = 1e-3
"".dec
#! 데이터셋 정의 함수
#! Chem.MolFromSmiles를 통해 	ClC1=CC=CC=C1 에서 Clc1ccccc1 로 바뀐다.
#! 77번째와 79번재는 뭘 하는지 모르겠음
def map_func(mol_str, cap):
    mol_str = mol_str.decode('utf-8')
    chem_mol = Chem.MolFromSmiles(mol_str[1:-1])
    img = Draw.MolToImage(chem_mol, size=(300,300))
    img = np.array(img)
    ### Fix color issue
    img[:,:,(2,1,0)] = img[:,:,(0,1,2)]
    # img = tf.io.read_file(image_path)
    img = tf.dtypes.cast(img, tf.float32)
#     img = tf.image.resize(img, (300, 300))
    return img, cap

def prep_func(image, cap):
    result_image = tf.keras.applications.inception_v3.preprocess_input(image)
    return result_image, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_val = dataset_val.batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


rdkit.Chem.MolFromSmiles('111')