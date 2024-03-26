import tkinter as tk
from tkinter import Text, Button, Entry

import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer




class Chatbot:
    def __init__(self):
        self.required_info = ["Country", "Local", "Industry Sector", "Gender", "Employee Type",
                              "Critical Risk", "Year", "Month", "Day", "Weekday", "Weekyear", "Season", "Is Holiday","Description"]
        self.user_info = ""
        self.current_intent_index = 0

    def start_conversation(self):
        self.user_info = ""
        self.current_intent_index = 0
        self.display_bot_message("Hi! I'm your chatbot. Let's get started.")
        self.ask_next_question()

    def ask_next_question(self):
        if self.current_intent_index < len(self.required_info):
            intent = self.required_info[self.current_intent_index]
            self.display_bot_message(f"Please provide your {intent}:")
        else:
            self.display_collected_info()

    def handle_user_input(self):
        user_input = self.UserInput.get().lower()  # Get user input
        self.UserInput.delete(0, tk.END)  # Clear the input field

        self.display_user_message(user_input)  # Display user input

        current_intent = self.required_info[self.current_intent_index]
        self.user_info += f"{current_intent} {user_input} "  # Append input to user_info string

        if self.current_intent_index < len(self.required_info) - 1:
            self.current_intent_index += 1
            self.ask_next_question()
        else:
            self.display_collected_info()  # Call display_collected_info even on final input

    def display_collected_info(self):
        collected_user_info = self.user_info  # Store user info for later display
        self.user_info = ""  # Reset user_info for next conversation

        self.display_bot_message("Here's the information you provided:")
        self.display_bot_message(collected_user_info)
        
        # Make prediction
        predicted_class = self.predict(collected_user_info)

        # Display prediction result
        self.display_bot_message(f"\nPredicted class: {predicted_class}")
        
        self.display_bot_message("\nYou can ask me anything else or type 'exit' to end the conversation.")

    def display_bot_message(self, message):
        self.ChatLog.config(state=tk.NORMAL)
        self.ChatLog.insert(tk.END, "Bot: " + message + '\n\n')
        self.ChatLog.config(state=tk.DISABLED)
        self.ChatLog.yview(tk.END)

    def display_user_message(self, message):
        self.ChatLog.config(state=tk.NORMAL)
        self.ChatLog.insert(tk.END, "You: " + message + '\n\n')
        self.ChatLog.config(state=tk.DISABLED)
        self.ChatLog.yview(tk.END)

    def create_widgets(self):
        self.master = tk.Tk()
        self.master.title("Accident Level Identifier: NLP Chatbot Project Tarang ")
        self.master.geometry("800x820")

        self.ChatLog = Text(self.master, bd=0, bg="white", height="20", width="70", font="Arial")
        self.ChatLog.config(state=tk.DISABLED)
        self.ChatLog.pack()

        self.UserInput = Entry(self.master, bd=0, bg="white", width="29", font="Arial")
        self.UserInput.pack(pady=2)
        self.UserInput.focus_set()

        self.SendButton = Button(self.master, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                                 bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                                 command=self.handle_user_input)
        self.SendButton.pack()

        # Add spacing between bot window and user input window
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)
    
    def predict(self, input_text):
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            loaded_tokenizer = pickle.load(f)
        
        
        #Load Model
        model = load_model('best_model.h5')
        self.display_bot_message('Load Model')
        
        #tokenizing the text
        tokenized_text= loaded_tokenizer.texts_to_sequences([input_text])

             
        #padding
        padded_text = pad_sequences(tokenized_text, maxlen=215, padding='pre', truncating='post')
        
        
        predictions= model.predict( padded_text)
        predicted_class= np.argmax(predictions)
        
        return predicted_class+1  # added 1 because label encoding starts from 0
        

    def run(self):
        self.create_widgets()
        self.start_conversation()
        self.master.mainloop()  # Added the mainloop call here


chatbot = Chatbot()
chatbot.run()
