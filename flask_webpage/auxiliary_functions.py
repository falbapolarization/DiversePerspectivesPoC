import json
import pickle
from openai import OpenAI
from tqdm import tqdm
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

def none_response(error_message):
    respose = {
        "error": error_message,
        "media article 1": None,
        "media article 2": None,
        "media bias 1": None,
        "media bias 2": None,
        "url article 1": None,
        "url article 2": None,
        "summery article 1": None,
        "summery article 2": None,
        "similarities": None,
        "differences": None,
    }
    return respose

#Similarity function
def cosine_similarity(vec1, vec2):
    return 1-distance.cosine(vec1, vec2)

#Is url:
def is_url(text):
    if text.startswith('http'):
        return True
    elif text.startswith('www'):
        return True
    else:
        return False

# load data:
def load_data(data_path):
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)
    return data

# Find input article
def find_input_article(input_article_url, data):
    input_article_indexes = data[data.article_url == input_article_url].index
    if len(input_article_indexes) == 0: #It is not in our data
        return -1 
    else:
        return input_article_indexes[0]

# Find similar article
def find_similar_article(input_article_embedding, data, bias):
    print("Finding most similar article.")
    best_similarity = 0
    most_similar_article_index = None
    for index, article in tqdm(data[data.media_bias == bias].iterrows()):
        similarity = cosine_similarity(input_article_embedding, article.summery_embedding)
        if best_similarity < similarity:
            best_similarity = similarity
            most_similar_article_index = index
    print("Most similar article found.")
    return most_similar_article_index, best_similarity

# GPT:

#following tutorial: https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/3/iterative
def get_completion(prompt, client, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response =  client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def generate_GPT_response(article_text_1, article_text_2, client):

    prompt = f"""
    You are an expert journalist. You are presented with two news article from different perspectives about the same topic.
    Article 1:
    ´´´
    {article_text_1}
    ´´´

    Article 2:
    ´´´
    {article_text_2}
    ´´´

    Generate a short summery for each article and list two similarities and two differences between the articles. 
    It should have a json format and the following 4 keys: 'summery article 1','summery article 2', 'similarities', 'differences'.
    """

    print("Generating GPT response.")
    response = get_completion(prompt, client)
    print("GPT response generated.")
    return response

# Parse and show output: 
def clean_list(text):
    if type(text) == list:
        return " ".join(text)
    else:
        return text

def parse_response(response, data, input_article_index, most_similar_article_index):
    try:
        parsed_response = json.loads(response)
        parsed_response['similarities'] = clean_list(parsed_response['similarities'])
        parsed_response['differences'] = clean_list(parsed_response['differences'])
        parsed_response['error'] = None
        parsed_response['media article 1'] = data.iloc[input_article_index].media_name
        parsed_response['media article 2'] = data.iloc[most_similar_article_index].media_name
        parsed_response['media bias 1'] = data.iloc[input_article_index].media_bias
        parsed_response['media bias 2'] = data.iloc[most_similar_article_index].media_bias
        parsed_response['url article 1'] = data.iloc[input_article_index].article_url
        parsed_response['url article 2'] = data.iloc[most_similar_article_index].article_url
    except:
        parsed_response = {
            'error': None, #"Parsing error.",
            'media article 1': data.iloc[input_article_index].media_name,
            'media article 2': data.iloc[most_similar_article_index].media_name,
            'media bias 1': data.iloc[input_article_index].media_bias,
            'media bias 2': data.iloc[most_similar_article_index].media_bias,
            'url article 1': data.iloc[input_article_index].article_url,
            'url article 2': data.iloc[most_similar_article_index].article_url,
            'summery article 1': data.iloc[input_article_index].article_text,
            'summery article 2': data.iloc[most_similar_article_index].article_text, 
            'similarities': "There has been a problem with GPT. Try later.", 
            'differences': "There has been a problem with GPT. Try later."
            }

    return parsed_response

# pipeline:
def pipeline(data_path, input_article, client, similarity_threshold = 0.1):
    data = load_data(data_path)
    if is_url(input_article):
        print("Input is a URL.")
        input_article_index = find_input_article(input_article, data)
        if input_article_index == -1: # Coudn't find input article in our database
            print("Input it is not an article in our database.")
            return none_response("Input is a url and the article is not in our database.")
        else: #the article is in our database
            print("It successfully found article in our database.")
            opposite_bias = "Left" if data.iloc[input_article_index].media_bias == "Right" else "Right"
            most_similar_article_index, best_similarity = find_similar_article(
                data.iloc[input_article_index].summery_embedding,
                data, 
                opposite_bias
            )
            if best_similarity< similarity_threshold:
                return  none_response("There is no similar article.")
            else:
                response = generate_GPT_response(
                    data.iloc[input_article_index].article_text, 
                    data.iloc[most_similar_article_index].article_text,
                    client)
                parsed_response = parse_response(response, data, input_article_index, most_similar_article_index)
                return parsed_response
    else: #input is a text.
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        input_embedding = model.encode(input_article)
        most_similar_article_index_Left, best_similarity_Left = find_similar_article(input_embedding, data, "Left")
        most_similar_article_index_Right, best_similarity_Right = find_similar_article(input_embedding, data, "Right")
        response = generate_GPT_response(
                    data.iloc[most_similar_article_index_Left].article_text, 
                    data.iloc[most_similar_article_index_Right].article_text,
                    client)
        parsed_response = parse_response(response, data, most_similar_article_index_Left, most_similar_article_index_Right)
        return parsed_response
