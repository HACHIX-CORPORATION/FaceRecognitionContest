# 標準ライブラリ

# 関連外部ライブラリ
from flask import Flask, render_template
from flask_uploads import configure_uploads
import ssl

# 内部ライブラリ
from uploads.route import uploads, files
from engine.route import engine
from api.api import api

# アプリケーションのインスタンス生成
app = Flask(__name__,
            instance_relative_config=True,
            static_folder="../Client/static",
            template_folder="../Client/templates")

# アプリケーションの設定
# sslの設定
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('cert.crt', 'server_secret_wo_pass.key')

app.config.from_pyfile('flask.cfg')

# ファイルをアップロードできるための設定
configure_uploads(app, files)

# ルートの設定
app.register_blueprint(uploads)
app.register_blueprint(engine)
app.register_blueprint(api)


ssl._create_default_https_context = ssl._create_unverified_context

# URLの設定
@app.route('/')
def main():
    return render_template('index.html')

# TODO: Duyとの確認が必要
# 確認内容：　以下の行のソースコードの意味は?　不必要であれば、削除しても良い？
# @app.route('/statistic')


if __name__ == "__main__":
    # アプリケーション開始
    # TODO: Trungとの確認が必要
    # 確認内容: threaded=Trueに変えた方が良い
    app.run(host='0.0.0.0', port=3000, ssl_context=context, threaded=False, debug=False)
