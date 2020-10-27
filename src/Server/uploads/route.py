# 標準ライブラリ
import os
import base64
import glob
import sys

# 関連外部ライブラリ
from flask import Blueprint
from flask_uploads import UploadSet, DATA, DOCUMENTS
from flask import jsonify, request

# 内部ライブラリ
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.facenet.facenet_engine import FacenetEngine
from database.db_utilities import DBUtility
from watcher.watcher import Watcher

# インスタンス生成
facenet = FacenetEngine()
db_util = DBUtility()
watcher = Watcher()

uploads = Blueprint('uploads', __name__)
files = UploadSet('files', DATA + DOCUMENTS)

SAVING_IMAGE_FOLDER_NAME = "images"

# デバッグモード
DEBUG_MODE = True


@uploads.route("/api/uploads", methods=['POST'])
def upload_binary():
    """
    Upload binary image API with format: <personName>_<image_source>
    And Post encode data to database
    """
    # 1: phan tich du lieu gửi từ web

    # 2: tạo encode (vector ) từ ảnh mặt

    # 3: lưu thông tin encode và thông tin staff lên database

    # 1 function: < 50 dòng (nhìn trong 1 màn hình)

    # 1 function chỉ nên làm 1 việc.

    # set img_dir
    cur_path = os.path.dirname(os.path.abspath(__file__))
    # Nên tránh magic number, magic string
    img_dir = os.path.join(cur_path, SAVING_IMAGE_FOLDER_NAME)

    if os.path.exists(img_dir) is False:
        os.mkdir(img_dir)

    try:
        # Read stream data
        data = request.stream.read()

        print("data = {}".format(data))

        # TODO: Duyとの確認
        # dataのフォーマットを確認し、コメントを追加する。
        # Get person_name and person_id from stream
        person_name = data[:data.find(b'_')]
        person_name = person_name.decode('utf-8')

        data = data[data.find(b'_') + 1:]
        person_id = data[:data.find(b'_')]
        person_id = person_id.decode('utf-8')
        # TODO: thay đổi /9
        data = data[data.find(b'/9'):]

        # 半角スペースと全角スペースを変換する。
        person_name = person_name.replace(" ", "-")
        person_name = person_name.replace("　", "-")
        person_id = person_id.replace(" ", "")
        person_id = person_id.replace("　", "")

        if DEBUG_MODE is True:
            print("name: {}, id: {}".format(person_name, person_id))

        # Decode image
        img_data = base64.b64decode(data)

        # make person dir
        person_dir_path = os.path.join(img_dir, person_name + "_" + person_id)
        if os.path.exists(person_dir_path) is False:
            os.mkdir(person_dir_path)

        # set image file name
        person_dir_path_files = glob.glob(person_dir_path + "/*")
        img_file_name = "{}.jpeg".format(person_name + "_" + person_id + str(len(person_dir_path_files)))
        img_file_path = os.path.join(person_dir_path, img_file_name)

        # Save image
        with open(img_file_path, "wb+") as g:
            g.write(img_data)
            g.close()

        # Make encode
        errcode, img_encode = facenet.make_encode(img_file_path)
        if errcode is 0:
            encode = img_encode.flatten()
            encode = encode.tolist()

            # Get staff info
            department = "null"
            staff_list = db_util.get_all_staff()
            for staff in staff_list:
                if staff['id'] == person_id:
                    department = staff['department']
                    person_name = staff['name']

            # Insert encode to database
            errcode, msg = db_util.insert_encode(name=person_name, id=person_id, department=department, encode=encode)

            response = jsonify({
                "errcode": errcode,
                "msg": msg
            })
        else:
            response = jsonify({
                "errcode": errcode,
                "msg": "写真には1つの顔のみを許可します。再度ご確認ください。"
            })

    except Exception as e:
        print(e)
        response = jsonify({
            "errcode": -1,
            "msg": "Error: {}".format(e)
        })

    return response
