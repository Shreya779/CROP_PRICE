import speech_recognition as sr
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"User: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

if __name__ == "__main__":
    speak("Hello! How can I assist you today?")

    while True:
        user_input = listen().lower()

        if "exit" in user_input:
            speak("Goodbye!")
            break


        elif "your_custom_command" in user_input:
            speak("Your custom response.")

        else:
            speak("I'm sorry, I didn't catch that. Can you please repeat?")
