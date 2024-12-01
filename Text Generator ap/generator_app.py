import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample text (your original text)
faqs = """I'm two weeks into a three-month tour and tonight 
I’m at a college in Chicago, performing a show at an on-campus bar in the basement of the student union. Three 
hundred people are packed into a space that should hold only half that many. The room is 
dark and noisy. The audience is on its way to being drunk—even very drunk—and while this 
may not be the worst performance environment I have ever faced, it’s close. In a gamble to 
take charge of the situation I’ve abandoned the small pipe-and-drape platform in the corner 
and now I’m standing on a table in the middle of the room. Every audience is different. 
Sometimes you have to charm them or cajole them, sometimes you have to entice or fascinate, 
and sometimes you have to roll up your sleeves and fight, winning the room with a careful 
blend of intensity and goodwill, convincing the audience that you’re either a genius or a 
madman and that, either way, they should probably stop for a second and listen.For the moment 
I’m fi lled with adrenaline and warm from victory. I am wide awake. Tomorrow I’ll leave early and travel all day so I can 
do it all over again in another town for another audience, but right now my thoughts are here in this room, and the room 
before that, and the hundreds and hundreds of theaters, auditoriums, and ballrooms before that, all the way back to my 
fi rst performance. I was nine years old. I made a coin vanish on the playground, the entire world went crazy, and I 
learned that you can say something with a magic trick that is hard to say any other way. PA RT O N E A L C H E M Y 
Som ewh e re i n my parents’ house there’s a picture of me at age seven. I’m crouched in the grass in the backyard on a 
summer evening, surrounded by fi refl ies, lifting my cupped hands as if I’m holding a secret and want to share it. 
At that age it’s easy to be amazed. The world is new and you are new in it and free from the ridiculous certainty that 
comes so easily with age that the inner workings of the universe are not only knowable but already known. My fi rst 
interest in magic came long before I became a magi cian, and though I have gone on to perform my show thou sands of 
times for hundreds of thousands of people, to this day when I think about magic I think about two memories from a time 
before I knew anything about tricks. The fi rst was when I lay on the fl oor under the piano when my dad played before 
bedtime. During the day he worked as a dentist, but we rarely saw him at the offi ce. We saw him when he came home and 
painted in the basement or paced the back yard with a yellow legal pad writing terse, fi ery letters to the editor of the
local paper about public policy and the environ ment. He’d read them to us at dinner and my mom would 12 Here Is Real 
Magic invariably protest—“Art! You can’t say that in public!”—and my younger brother and I would laugh in delight at
her exas peration. But in the evenings he would turn out the lights and sit at the piano and I would lie underneath, 
listening. The only light in the room came from the lamp for the sheet music— Beethoven, Bach, Rachmaninoff ."""

# Tokenization and preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
input_sequences = []
for sentence in faqs.split("\n"):
    tokenize_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenize_sentence)):
        input_sequences.append(tokenize_sentence[:i + 1])

max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

x = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
num_classes = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=num_classes)

# Building the LSTM model
model = Sequential()
model.add(Embedding (num_classes, 100, input_length=max_len - 1))
model.add(LSTM(200))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=50)

# Streamlit user interface
st.title("Magic Text Generator")
st.write("This application generates text based on a magician's narrative. Enter a prompt to see the magic unfold!")

# Text customization options
font_selection = st.selectbox("Select Font:", ["Arial", "Courier New", "Georgia", "Times New Roman"])
text_style = st.selectbox("Select Text Style:", ["Normal", "Bold", "Italic", "Underline"])
text_alignment = st.selectbox("Select Text Alignment:", ["Left", "Center", "Right"])

# User input
input_text = st.text_input("Enter a starting phrase:", "I’m at a college")

if st.button("Generate"):
    generated_text = input_text
    for i in range(15):  # Generate 15 words
        token_text = tokenizer.texts_to_sequences([generated_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
        pos = np.argmax(model.predict(padded_token_text), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == pos:
                generated_text += " " + word
                break

    # Display generated text with customization
    st.write("Generated Text:")
    if text_style == "Bold":
        st.markdown(f"<p style='font-family:{font_selection}; font-weight:bold; text-align:{text_alignment.lower()};'>{generated_text}</p>", unsafe_allow_html=True)
    elif text_style == "Italic":
        st.markdown(f"<p style='font-family:{font_selection}; font-style:italic; text-align:{text_alignment.lower()};'>{generated_text}</p>", unsafe_allow_html=True)
    elif text_style == "Underline":
        st.markdown(f"<p style='font-family:{font_selection}; text-decoration:underline; text-align:{text_alignment.lower()};'>{generated_text}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-family:{font_selection}; text-align:{text_alignment.lower()};'>{generated_text}</p>", unsafe_allow_html=True)