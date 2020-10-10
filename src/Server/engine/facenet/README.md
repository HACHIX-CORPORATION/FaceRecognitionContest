# Omni Face Recognition engine

## Finding threshold 
```
python3 threshold.py --path "../../dataset/japanese/"
```

## Test on Japanese dataset
The following two samples using compresses data from test/compress

```
cd test 
python3 test_classifier.py
```
Changing threshold for manual classifier (Class Baseline). Line 115, 
```
threshold = 0.95
```

## Test on Celebrity dataset
```
cd test 
python3 test_facenet_engine.py
```

## Test on embedding file
This file make a compressed data of embedding vectors from japanese dataset and save inside compressed folder
```
cd test
python3 test_embedding.py --path "../../../dataset/japanese/"
```


## Solutions from AiVN Celebrity Face Recognition 
### 1st Solution 
https://bitbucket.org/dungnb1333/dnb-facerecognition-aivivn/src/master/

### 2nd Solution 
https://forum.machinelearningcoban.com/t/aivivn-face-recognition-2nd-solution/4730/3

### 3rd Solution 
https://forum.machinelearningcoban.com/t/aivivn-face-recognition-3nd-solution/4736
Điểm đặc biệt của giải pháp này là không cần thêm models để train classification. Chỉ dựa vào cách chọn threshold distances. Cách chọn được phân tích cụ thể: \
https://www.kaggle.com/suicaokhoailang/aivivn-celebs-re-identification-baseline/?fbclid=IwAR0L7uDaqEK7PDoaRYQX01vRJSDAJ_dv3YAI5lqxPcavAwWaiVh8rOmKeA8

## Issue
After fixing bug, there will be a warning of optimization CPU setting
![Image of Issue](https://i.ibb.co/K5BnGY5/Screen-Shot-2020-01-31-at-13-04-49.png)

(2020/01/18)
```
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/."
```

Follow https://github.com/dmlc/xgboost/issues/1715
```
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

# facenet評価実行
python facenet_eval.py --data_set '../../dataset/japanese/' --model_path './model/keras-facenet/model/facenet_keras.h5' --using_SVM 1
python facenet_engine.py --data_set '../../dataset/train/japanese/' --model_path './model/keras-facenet/model/facenet_keras.h5' --classifier_filename './model/SVM_classifer.pkl'
