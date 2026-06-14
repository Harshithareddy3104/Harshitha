import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from wordcloud import WordCloud

# Global variables
model = None
tokenizer = None
max_len = 100
model_file = "hotel_review_model.h5"
tokenizer_file = "tokenizer.pkl"
df = None
prediction_history = []

def load_dataset():
    global df
    file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=(("CSV Files", "*.csv"),))
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Dataset Loaded", f"Dataset loaded with {len(df)} rows")
        dataset_info_label.config(text=f"Rows: {len(df)} | Columns: {', '.join(df.columns)}")
    else:
        messagebox.showerror("Error", "No file selected")

def show_analytics():
    if df is None:
        messagebox.showerror("Error", "Load dataset first")
        return

    # Pie chart for sentiment distribution
    plt.figure(figsize=(5, 4))
    df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'orange', 'red'])
    plt.title("Sentiment Distribution")
    plt.show()

    # Bar chart for business impact
    plt.figure(figsize=(5, 4))
    df['BusinessImpact'].value_counts().plot(kind='bar', color=['blue', 'purple', 'pink'])
    plt.title("Business Impact Distribution")
    plt.xlabel("Impact")
    plt.ylabel("Count")
    plt.show()

    # Word cloud
    text = " ".join(df['Review'].astype(str))
    wc = WordCloud(width=500, height=400, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.show()

def train_model():
    global model, tokenizer
    if df is None:
        messagebox.showerror("Error", "Load dataset first")
        return

    X = df['Review'].astype(str)
    y = df['BusinessImpact'].map({'Low': 0, 'Medium': 1, 'High': 2})

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len)

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=3, batch_size=4, validation_data=(X_test, y_test))

    # Save model and tokenizer
    model.save(model_file)
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

    messagebox.showinfo("Training Complete", "Model trained and saved successfully!")

    # Plot accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

def load_trained_model():
    global model, tokenizer
    if os.path.exists(model_file) and os.path.exists(tokenizer_file):
        model = load_model(model_file)
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
        messagebox.showinfo("Model Loaded", "Trained model loaded successfully!")
    else:
        messagebox.showerror("Error", "No saved model found. Please train the model first.")

def predict_review():
    global model, tokenizer, prediction_history
    if model is None or tokenizer is None:
        messagebox.showerror("Error", "Load or train the model first")
        return

    review = review_entry.get("1.0", tk.END).strip()
    if not review:
        messagebox.showerror("Error", "Please enter a review")
        return

    seq = tokenizer.texts_to_sequences([review])
    pad_seq = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad_seq)
    impact_idx = np.argmax(pred)

    if impact_idx == 0:
        impact = "Low Business Impact"
    elif impact_idx == 1:
        impact = "Medium Business Impact"
    else:
        impact = "High Business Impact"

    prediction_history.append((review, impact))
    update_history_table()
    messagebox.showinfo("Prediction Result", f"Predicted Impact: {impact}")

def update_history_table():
    for row in history_tree.get_children():
        history_tree.delete(row)
    for i, (review, impact) in enumerate(prediction_history, start=1):
        history_tree.insert("", "end", values=(i, review[:30]+"...", impact))

# Tkinter GUI with Tabs
root = tk.Tk()
root.title("Hotel Review Analysis with Deep Learning")
root.geometry("800x600")

tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

tab_control.add(tab1, text="Train Model")
tab_control.add(tab2, text="Analytics")
tab_control.add(tab3, text="Prediction")
tab_control.pack(expand=1, fill="both")

# Tab 1: Training
tk.Label(tab1, text="hotel review analysis for the prediction of bussinessusinf deep learning approch", font=("Arial", 18, "bold")).pack(pady=10)
btn_load = tk.Button(tab1, text="Load Dataset", command=load_dataset, width=20, bg="#4caf50", fg="white", font=("Arial", 12))
btn_load.pack(pady=5)
dataset_info_label = tk.Label(tab1, text="", font=("Arial", 12))
dataset_info_label.pack(pady=5)
btn_train = tk.Button(tab1, text="Train & Save Model", command=train_model, width=20, bg="#2196f3", fg="white", font=("Arial", 12))
btn_train.pack(pady=5)
btn_load_model = tk.Button(tab1, text="Load Saved Model", command=load_trained_model, width=20, bg="#9c27b0", fg="white", font=("Arial", 12))
btn_load_model.pack(pady=5)

# Tab 2: Analytics
tk.Label(tab2, text="Analytics & Visualization", font=("Arial", 18, "bold")).pack(pady=10)
btn_analytics = tk.Button(tab2, text="Show Analytics", command=show_analytics, width=20, bg="#ff9800", fg="white", font=("Arial", 12))
btn_analytics.pack(pady=10)

# Tab 3: Prediction
tk.Label(tab3, text="Enter Review for Prediction", font=("Arial", 18, "bold")).pack(pady=10)
review_entry = tk.Text(tab3, height=4, width=60)
review_entry.pack(pady=10)
btn_predict = tk.Button(tab3, text="Predict", command=predict_review, width=20, bg="#ff5722", fg="white", font=("Arial", 12))
btn_predict.pack(pady=10)

# Prediction History Table
history_tree = ttk.Treeview(tab3, columns=("SNo", "Review", "Impact"), show="headings")
history_tree.heading("SNo", text="S.No")
history_tree.heading("Review", text="Review")
history_tree.heading("Impact", text="Business Impact")
history_tree.pack(pady=10, fill="both")

root.mainloop()