# 画像分類
from flask import Blueprint,render_template,request,url_for,redirect,flash,abort
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import numpy as np
import os
from flask import Flask, request, redirect, render_template, flash
from company_blog.main.forms import ClassificationForm
from company_blog.models import BlogCategory,BlogPost,Inquiry,Classification

classify = Blueprint('classification', __name__)

classes = ["ビロウドモウズイカ","エノコログサ","ヒメジョン","ヒメオドリコソウ","オオイヌノフグリ","オオキンケイギク","シロツメクサ","スギナ","タンポポ","ヤグルマギク"]
texts = ["ビロウドモウズイカの説明","エノコログサの説明","ヒメジョンの説明","ヒメオドリコソウの説明","オオイヌノフグリの説明","オオキンケイギクの説明","シロツメクサの説明","スギナの説明","タンポポの説明","ヤグルマギクの説明"]

image_size = 224

UPLOAD_FOLDER = r"E:\weedblog\company_blog\static\uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model(r'E:\weedblog\company_blog\epoc30.h5',compile=False)#学習済みモデルをロード


@classify.route('/judge', methods=['GET', 'POST'])
def judge():
    form = ClassificationForm()
    if request.method == 'POST':
        if 'input_image' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['input_image']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            image_url = url_for('static', filename=f'uploads/{filename}')

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, color_mode='rgb', target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])

            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "この雑草は " + classes[predicted] + " です"
            pred_text = texts[predicted]
            return render_template("classification.html",answer=pred_answer,text=pred_text,form=form,image_post=image_url)
    return render_template("classification.html",answer="",form=form)