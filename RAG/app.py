from flask import Flask, render_template, request, url_for, redirect, session
from pymongo import MongoClient
import bcrypt
import ollama
from flask import Flask , request , render_template
import os
import json
import numpy as np
from numpy.linalg import norm
import logging


app = Flask(__name__)
#encryption relies on secret keys so they could be run
app.secret_key = "testing"

# #connect to your Mongo DB database
def MongoDB():
    client = MongoClient("mongodb://127.0.0.1:27017")
    db = client.get_database('Fitness_guru')
    records = db.register
    return records
# records = MongoDB()


##Connect with Docker Image###
def dockerMongoDB():
    client = MongoClient(host='localhost',
                            port=27017)
    db = client.Fitness_guru
    # pw = "test123"
    # hashed = bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt())
    records = db.register
    # records.insert_one({
    #     "name": "Test Test",
    #     "email": "test@yahoo.com",
    #     "password": hashed
    # })
    return records

records = dockerMongoDB()

# open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings


# find cosine similarity of every chunk to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


#assign URLs to have a particular route 
@app.route("/", methods=['post', 'get'])
def index():
    message = ''
    #if method post in index
    if "email" in session:
        return redirect(url_for("chatbot"))
    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        #if found in database showcase that it's found 
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('index.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('index.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('index.html', message=message)
        else:
            #hash the password and encode it
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            #assing them in a dictionary in key value pairs
            user_input = {'name': user, 'email': email, 'password': hashed}
            #insert it in the record collection
            records.insert_one(user_input)
            
            #find the new created account and its email
            user_data = records.find_one({"email": email})
            new_email = user_data['email']
            #if registered redirect to logged in as the registered user
            return render_template('chatbot.html', email=new_email)
    return render_template('index.html')

@app.route("/login", methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect(url_for("chatbot"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        #check if email exists in database
        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            #encode the password and check if it matches
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('chatbot'))
            else:
                if "email" in session:
                    return redirect(url_for("chatbot"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)

@app.route('/logged_in')
def logged_in():
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', email=email)
    else:
        return redirect(url_for("login"))

@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('index.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():

    query_input = None
    response = None
    
    SYSTEM_PROMPT = """If you're unsure, just say that you don't know.
        The text or context that you have been provided is called as the 'fitness database'.
        Context:
    """
    # # open file
    filename = "workouts.txt"
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
    if request.method == 'POST':
        query_input = request.form.get('query-input')

        # prompt = input("what do you want to know? -> ")
        # strongly recommended that all embeddings are generated by the same model (don't mix and match)
        prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query_input)["embedding"]
        # find most similar to each other
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        
        if query_input:
            try:
                response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
                    },
                    {"role": "user", "content": query_input},
                ],
            )
                logging.info(f"Response from Ollama:{response}")
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                output = "Sorry, an error occurred while processing your request."
    return render_template('chatbot.html', query_input=query_input, output=response)
        # print("\n\n")
        # print(response["message"]["content"])


if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=5000)