import random
import pyttsx3
import tkinter as tk
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

class MotivationalChatbotGUI:
    """A GUI for the Motivational Chatbot."""

    def __init__(self, master):
        self.master = master
        self.master.title("Motivational Chatbot")

        self.chatbot = MotivationalChatbot()

        self.label = tk.Label(master, text="Enter your message and press Enter:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(master, width=30)
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", self.process_input)

        self.text_display = tk.Text(master, width=40, height=5)
        self.text_display.pack(pady=5)

    def process_input(self, event):
        user_input = self.entry.get().lower()
        response = self.chatbot.generate_response(user_input)
        self.text_display.insert(tk.END, f"Bot: {response}\n\n")
        self.chatbot.speak(response)

class MotivationalChatbot:
    """A chatbot that provides motivational statements, jokes, and help responses."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.model = load_model('chatbot_model.h5')
        self.words = []
        self.classes = []
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()

    def load_intents(self):
        """Load intents from JSON file."""
        # Replace this with your own code to load intents from a JSON file
        intents = {
            "intents": [
                {"tag": "greeting", "patterns": ["hello", "hi", "hey"], "responses": ["Hello!", "Hi there!"]},
                {"tag": "joke", "patterns": ["tell me a joke", "joke", "jokes"], "responses": ["Here's a joke: ..."]},
                {"tag": "motivation", "patterns": ["motivate me", "motivational", "inspire me"], "responses": ["You can do it!", "Believe in yourself!"]}
            ]
        }
        return intents

    def preprocess_input(self, input_text):
        """Tokenize and preprocess input text."""
        words = nltk.word_tokenize(input_text)
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def generate_response(self, input_text):
        """Generate response based on user input."""
        words = self.preprocess_input(input_text)
        bag = [0]*len(self.words)
        for word in words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1

        result = self.model.predict(np.array([bag]))[0]
        threshold = 0.25
        tag = self.classes[np.argmax(result)]
        if result[np.argmax(result)] > threshold:
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
        else:
            return "I'm sorry, I don't understand that."

    def speak(self, text):
        """Use text-to-speech to speak the provided text."""
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotivationalChatbotGUI(root)
    root.mainloop()
