"""
Api for post and get staff and encode data from client
"""
# 外部ライブラリ
from flask import Blueprint
from flask import jsonify, request

# 内部モジュール
from database.db_utilities import DBUtility

api = Blueprint('api', __name__)
db_util = DBUtility()


@api.route("/api/staff", methods=['GET', 'POST'])
def staff():
    """
    get and post staff information

    Api: /api/staff?item=info
    """
    if request.method == 'GET':
        item = request.args.get('item')
        if item == "all":
            # get staff info from DB
            try:
                res = db_util.get_all_staff()
                response = (jsonify(res), 200)
            except Exception as e:
                print(e)
                response = ('Error getting staff info', 500)
        else:
            response = ("Unknown query!", 400)

    else:
        response = ('Post method not supported', 500)

    return response


@api.route("/api/encode", methods=['GET', 'POST'])
def encode():
    """
    get and post encode

    GET: /api/encode?item=all ==> for get all encode
        /api/encode?item=<id> ==> for get encode of one id
    POST: /api/encode
        body: {
           "_id": str(self.today),
            "id": int(self.today),
            "name": "hachixxx",
            "department": "development01",
            "encode": [1, 2, 3, 4]
        }

    """
    if request.method == 'GET':
        item = request.args.get("item")
        if item == "all":
            try:
                res = db_util.get_encode({})
                response = (jsonify(list(res)), 200)
            except:
                response = ('Error getting all encode info', 500)
        else:
            try:
                id = int(item)
                res = db_util.find_one_encode(id)
                response = (jsonify(res), 200)
            except:
                response = ('Error getting one encode info', 500)

    else:
        data = request.get_data()
        try:
            errcode, msg = db_util.insert_encode(data)
            response = (jsonify(msg), 200)
        except:
            response = ('Error posting encode', 500)
    return response




