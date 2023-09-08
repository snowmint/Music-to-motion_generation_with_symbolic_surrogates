from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import sys
import os
import json

app = Flask(__name__, template_folder='templates',
            static_folder='static', static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST", "GET"])
def upload():
    
    if request.method == "POST":
        result_json_string = request.form["json_result_text"]

        with open("result.txt", 'a') as f:
            f.write(result_json_string)
            f.write('\n')
            f.write('\n')
        
        print(result_json_string)
    
    msg = "Status 200: 您已完成填寫並成功上傳問卷，謝謝您"
    return msg
    #return jsonify({'htmlresponse': render_template('response.html', msg=msg)})

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)
