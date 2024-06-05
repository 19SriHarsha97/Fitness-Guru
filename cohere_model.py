import pandas as pd
import cohere
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables 
load_dotenv()  
cohere_api_key = os.environ["COHERE_API_KEY"]
co = cohere.Client(cohere_api_key)

# --- Dataset Loading (Adapt This!) ---
def load_exercise_data(csv_file):
    df = pd.read_csv(csv_file)
    # ... potentially extract relevant columns & data cleaning ... 
    return df 

# Replace 'your_data.csv' with your actual filename
exercise_data = load_exercise_data('megaGymDataset.csv') 

# ---  Process User Queries ---
def gather_user_preferences():
    goal = st.selectbox("What's your main fitness goal?", 
                        ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"])
    experience = st.radio("What's your experience level?",
                          ["Beginner", "Intermediate", "Advanced"])
    body_part = st.multiselect("Which body part do you want to focus on?",
                             ["Chest", "Back", "Quadriceps","Hamstrings", "Biceps","Triceps","Forearms", "Shoulders", "Full Body"])
    equipment = st.selectbox("What equipment do you have access to?",
                             [ "Dumbbell", "Barbell", "Machine", "Bands", "Body Only","All"])
    restrictions = st.checkbox("Any injuries or limitations?")
    return {
        "goal": goal,
        "experience": experience,
        "body_part": body_part,
        "equipment": equipment,
        "restrictions": restrictions
    }

def filter_exercises(data, experience, body_parts, equipment):
    top_n = 6 if len(body_parts)==1 else 3
    filtered_data = pd.DataFrame()
    
    # Iterate over each selected body part
    for body_part in body_parts:
        # Filter the data for the current body part
        body_part_filtered = data[
            (data['Level'] == experience) &
            (data['BodyPart'].str.contains(body_part, case=False) if 'Full Body' not in body_parts else True) &
            (data['Equipment'].str.contains(equipment, case=False) if equipment!='All' else True) 
        ]
        body_part_filtered = body_part_filtered.sort_values(by='Rating',ascending=False)
        # Concatenate the filtered results with the overall filtered data
        filtered_data = pd.concat([filtered_data, body_part_filtered[:top_n]])

    # filtered_data = filtered_data.sort_values(by='Rating', ascending=False)
    return filtered_data

def format_filtered_exercises(filtered_exercises):
    result = "Here are the top recommended exercises based on your preferences:\n\n"
    for _, row in filtered_exercises.iterrows():
        result += f"Exercise: {row['Title']}\n"
        result += f"Description: {row['Desc']}\n"
        result += f"Equipment: {row['Equipment']}\n"
        result += f"Body Part: {row['BodyPart']}\n"
        result += f"Rating: {row['Rating']}\n\n"
    return result

def process_query(query, exercise_data, user_preferences=None):
    if user_preferences is None:
        user_preferences = gather_user_preferences()
        return process_query(query, exercise_data, user_preferences)
    
    # Check if the user is asking about a specific exercise
    if user_asks_about_exercise(query):
        exercise_name = extract_exercise_name(query)
        exercise_description = describe_exercise(exercise_name, exercise_data)
        return exercise_description
    else:
        # Filter exercises based on user preferences
        filtered_exercises = filter_exercises(
            exercise_data, 
            user_preferences['experience'], 
            user_preferences['body_part'], 
            user_preferences['equipment']
        )
        
        # If no exercises found, return a message
        if filtered_exercises.empty:
            return "Sorry, no matching exercises found."
        
        # Format the filtered exercises
        formatted_exercises = format_filtered_exercises(filtered_exercises)
        
        # Generate a detailed workout plan using Cohere
        prompt = craft_fitness_prompt(query, formatted_exercises)
        response = co.generate( 
            model='command-nightly',  
            prompt=prompt,   
            stop_sequences=["--"]) 
        return response.generations[0].text

# --- Helper Functions (You might need to adjust) ---
def user_asks_about_exercise(query):
    # Simple keyword detection, make this smarter!
    return "describe" in query or "how to" in query 

def extract_exercise_name(query):
    # Basic extraction, improve this with NLP techniques if needed
    return query.split("describe ")[1] 

def describe_exercise(exercise, data):
    # Lookup exercise in 'data' & construct a description
    match = data[data['Title'].str.contains(exercise, case=False, na=False)]
    if not match.empty:
        row = match.iloc[0]
        description = (f"Exercise: {row['Title']}\n"
                       f"Description: {row['Desc']}\n"
                       f"Equipment: {row['Equipment']}\n"
                       f"Body Part: {row['BodyPart']}\n"
                       f"Rating: {row['Rating']}\n")
        return description
    else:
        return "Sorry, exercise not found."

def craft_fitness_prompt(query, formatted_exercises):
    return (f"You are a fitness expert. A user asked the following question: {query}.\n"
            f"Based on their preferences, here are some recommended exercises:\n\n{formatted_exercises}"
            "Please create a detailed workout plan based on the above exercises.")

# --- Streamlit UI ---
st.title("Fitness Guru")

# Gather preferences right at the start 
user_preferences = gather_user_preferences() 

user_input = st.text_input("Ask me about workouts or fitness...")

if st.button("Submit"): 
    chatbot_response = process_query(user_input, exercise_data, user_preferences)
    st.write("Chatbot:", chatbot_response)

