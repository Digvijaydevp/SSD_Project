from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Device configuration (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to saved files
CSV_FILE_PATH = "./test_predictions_after_finetuning.csv"
MODEL_PATH = "final_model3.pth"
METRICS_FILE = "training_history.json"
PERFORMANCE_METRICS_FILE = "performance_metrics.json"
CLASS_DISTRIBUTION_FILE = "class_distribution.json"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Clean tweet function
def clean_tweet(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

labels = ["negative", "neutral", "positive"]

@app.route("/custom-text", methods=["POST"])
def custom_text():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Clean and tokenize text
    cleaned_text = clean_tweet(text)
    inputs = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=80,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_sentiment = labels[predicted_class]

    print(predicted_sentiment)
    return jsonify({
        "text": text,
        "predicted_sentiment": predicted_sentiment
    })

@app.route("/hashtag", methods=["POST"])
def hashtag():
    data = request.get_json()
    keyword = data.get("input")
    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        filtered_data = df[df['text'].str.contains(keyword, case=False, na=False)]
        
        filtered_data['predicted_label'] = filtered_data['predicted_label'].str.lower()
        
        # Group tweets by sentiment
        positive_tweets = filtered_data[filtered_data['predicted_label'] == 'positive'][['text', 'predicted_label']].rename(columns={"predicted_label": "sentiment"}).to_dict(orient="records")
        negative_tweets = filtered_data[filtered_data['predicted_label'] == 'negative'][['text', 'predicted_label']].rename(columns={"predicted_label": "sentiment"}).to_dict(orient="records")
        neutral_tweets = filtered_data[filtered_data['predicted_label'] == 'neutral'][['text', 'predicted_label']].rename(columns={"predicted_label": "sentiment"}).to_dict(orient="records")
        
        # Calculate overall sentiment
        total_tweets = len(filtered_data)
        positive_count = len(positive_tweets)
        negative_count = len(negative_tweets)
        neutral_count = len(neutral_tweets)
        
        # Determine overall sentiment
        if positive_count >= negative_count and positive_count >= neutral_count:
            overall_sentiment = 'Positive'
            overall_percentage = f"{positive_count}/{total_tweets}"
        elif negative_count >= positive_count and negative_count >= neutral_count:
            overall_sentiment = 'Negative'
            overall_percentage = f"{negative_count}/{total_tweets}"
        else:
            overall_sentiment = 'Neutral'
            overall_percentage = f"{neutral_count}/{total_tweets}"
        
        return jsonify({
            "total": total_tweets,
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "overall_sentiment": {
                "sentiment": overall_sentiment,
                "percentage": overall_percentage
            },
            "tweets": {
                "positive": positive_tweets,
                "negative": negative_tweets,
                "neutral": neutral_tweets
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reports", methods=["GET"])
def reports():
    try:
        with open(PERFORMANCE_METRICS_FILE, "r") as metrics_file:
            performance_metrics = json.load(metrics_file)
        
        with open(CLASS_DISTRIBUTION_FILE, "r") as distribution_file:
            class_distribution = json.load(distribution_file)
        
        # Load training info
        with open(METRICS_FILE, "r") as training_file:
            training_info = json.load(training_file)

        return jsonify({
            "performance_metrics": performance_metrics,
            "class_distribution": class_distribution,
            "training_info": training_info
        })
    except FileNotFoundError:
        return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
