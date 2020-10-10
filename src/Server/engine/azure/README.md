# Azure Face API 

## 事前準備
1. secret.pyのファイルをAPIのフォルダに入れる。
2. secret.pyのファイルは以下のようになっている。 
```
CONFIG = {
    "FACE_SUBSCRIPTION_KEY": 'YOUR_KEY',
    "FACE_ENDPOINT": "https://westcentralus.api.cognitive.microsoft.com/"
}
```

## 実行
cd engine\API 
python azure_eval.py --data_set ../../dataset/japanese

## 画像検知のみを実行する
```
python3 detect-face.py
```

## Issue 

```
  File "detect-face.py", line 34, in <module>
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url)
  File "/Users/admin/opt/anaconda3/lib/python3.7/site-packages/azure/cognitiveservices/vision/face/operations/_face_operations.py", line 549, in detect_with_url
    raise models.APIErrorException(self._deserialize, response)
azure.cognitiveservices.vision.face.models._models_py3.APIErrorException: (404) Resource not found
```
### Fixed
When setting up the endpoint for environment, remember to not inclue "/face/v1.0".

