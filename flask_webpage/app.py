#python version 3.10.11
from openai import OpenAI
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from auxiliary_functions import pipeline

global_var = {}
global_var["data_path"] = "/Users/falbanese/Documents/OtherProjects/PolarizationCHP/CollectAndProcessData/Data/articles_3JUN_processed.pickle"
global_var["openAI_key"] = ""
global_var["client"] = None

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    data_path = global_var["data_path"]
    
    char_count = None
    response = None
    error_message = None
    media_article_1 = None
    media_article_2 = None
    media_bias_1 = None
    media_bias_2 = None
    url_article_1 = None
    url_article_2 = None
    summery_article_1 = None
    summery_article_2 = None
    similarities = None
    differences = None
    input_text = None
    if request.method == 'POST':
        input_text = request.form['input']
        char_count = len(input_text)
        response = pipeline(data_path, input_text, global_var["client"])
        error_message = response["error"]
        media_article_1 = response['media article 1']
        media_article_2 = response['media article 2']
        media_bias_1 = response['media bias 1']
        media_bias_2 = response['media bias 2']
        url_article_1 = response['url article 1']
        url_article_2 = response['url article 2']
        summery_article_1 = response['summery article 1']
        summery_article_2 = response['summery article 2']
        similarities = response['similarities']
        differences = response['differences']
    return render_template(
        'index.html', 
        char_count=char_count, 
        input_text=input_text,
        response=response, 
        error_message=error_message,
        media_article_1 = media_article_1,
        media_article_2 = media_article_2,
        media_bias_1 = media_bias_1,
        media_bias_2 = media_bias_2,
        url_article_1 = url_article_1,
        url_article_2 = url_article_2,
        summery_article_1 = summery_article_1,
        summery_article_2 = summery_article_2,
        similarities = similarities,
        differences = differences,
        )

@app.route('/<path:input_url>', methods=['GET', 'POST'])
def webpage(input_url: str):
    print("Input URL:", input_url)
    
    data_path = global_var["data_path"]
    
    input_text = input_url
    char_count = len(input_text)
    response = pipeline(data_path, input_text, global_var["client"])
    error_message = response["error"]
    media_article_1 = response['media article 1']
    media_article_2 = response['media article 2']
    media_bias_1 = response['media bias 1']
    media_bias_2 = response['media bias 2']
    url_article_1 = response['url article 1']
    url_article_2 = response['url article 2']
    summery_article_1 = response['summery article 1']
    summery_article_2 = response['summery article 2']
    similarities = response['similarities']
    differences = response['differences']
    if request.method == 'POST':
        input_text = request.form['input']
        char_count = len(input_text)
        response = pipeline(data_path, input_text, global_var["client"])
        error_message = response["error"]
        media_article_1 = response['media article 1']
        media_article_2 = response['media article 2']
        media_bias_1 = response['media bias 1']
        media_bias_2 = response['media bias 2']
        url_article_1 = response['url article 1']
        url_article_2 = response['url article 2']
        summery_article_1 = response['summery article 1']
        summery_article_2 = response['summery article 2']
        similarities = response['similarities']
        differences = response['differences']
    return render_template(
        'index.html', 
        char_count=char_count, 
        input_text=input_text,
        response=response, 
        error_message=error_message,
        media_article_1 = media_article_1,
        media_article_2 = media_article_2,
        media_bias_1 = media_bias_1,
        media_bias_2 = media_bias_2,
        url_article_1 = url_article_1,
        url_article_2 = url_article_2,
        summery_article_1 = summery_article_1,
        summery_article_2 = summery_article_2,
        similarities = similarities,
        differences = differences,
        )

@app.route('/Keys/<key>', methods=['GET', 'POST'])
def OpenAIKeys(key: str):
    global_var["openAI_key"] = key
    global_var["client"]  = OpenAI(
        api_key = "sk-"+key,
    )
    return ""


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
