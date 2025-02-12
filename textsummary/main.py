import pandas as pd
import matplotlib
import base64
from io import BytesIO
import os
from flask import Flask, render_template, request, jsonify, url_for
import mysql.connector
from transformers import pipeline
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# Load the fine-tuned model
from transformers import BartForConditionalGeneration, BartTokenizer


fine_tuned_model_path = r'./fine_tuned_bart'
fine_tuned_model = BartForConditionalGeneration.from_pretrained(
    fine_tuned_model_path)
fine_tuned_tokenizer = BartTokenizer.from_pretrained(fine_tuned_model_path)

# Function to summarize using your fine-tuned model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fine_tuned_model_pipeline = pipeline(
    "summarization", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, device=device)


def generate_summary_fine_tuned(text, max_length=100, min_length=25):
    inputs = fine_tuned_tokenizer(
        text, return_tensors="pt", max_length=1024, truncation=True)
    if len(inputs['input_ids'][0]) > 1024:  # Ensure the input doesn't exceed token limit
        return "Text too long for fine-tuned model.", 400
    summary_ids = fine_tuned_model.generate(
        inputs["input_ids"], max_length=max_length, min_length=min_length)
    return fine_tuned_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Configure Matplotlib for non-GUI environments
matplotlib.use('Agg')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'news_db',
}


# Establish database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None


# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Define the device here
sentiment_analyzer = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Sentiment labels
SENTIMENT_LIST = ["positive", "negative", "neutral",
                  "joy", "sadness", "anger", "fear", "trust"]


# Function to generate sentiment analysis plot and save to CSV
def generate_sentiment_plot_and_save_to_csv(text, summary):
    try:
        # Perform sentiment analysis
        analysis_output = sentiment_analyzer(
            text, candidate_labels=SENTIMENT_LIST, multi_label=True)
        sentiments = {emotion: score for emotion, score in zip(
            analysis_output['labels'], analysis_output['scores'])}

        # Generate the sentiment plot
        df_sentiments = pd.DataFrame(
            sentiments.items(), columns=['Emotion', 'Score'])

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        sns.barplot(x='Emotion', y='Score', data=df_sentiments)

        plt.title('Sentiment Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        sns.despine()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('ascii')

        # Prepare data for CSV
        sentiment_data = {
            "Text Summary": summary,
            "Positive Score": sentiments.get("positive", 0),
            "Negative Score": sentiments.get("negative", 0),
            "Neutral Score": sentiments.get("neutral", 0),
            "Joy Score": sentiments.get("joy", 0),
            "Sadness Score": sentiments.get("sadness", 0),
            "Anger Score": sentiments.get("anger", 0),
            "Fear Score": sentiments.get("fear", 0),
            "Trust Score": sentiments.get("trust", 0),
            # Saving all emotions as a string
            "Other Emotions": str(sentiments)
        }

        # Save to CSV
        sentiment_df = pd.DataFrame([sentiment_data])

        # If the CSV file doesn't exist, create it with header
        file_exists = os.path.exists("sentiment_results.csv")
        sentiment_df.to_csv("sentiment_results.csv", mode='a',
                            header=not file_exists, index=False)

        return plot_base64
    except Exception as e:
        print(f"Error generating sentiment plot and saving to CSV: {e}")
        return None


# @app.route('/')
# def index():
    connection = get_db_connection()
    if not connection:
        return "Failed to connect to the database.", 500

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, title, content FROM news ORDER BY id DESC LIMIT 10")
        results = cursor.fetchall()
        return render_template('index.html', news=results)
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return "An error occurred while fetching data.", 500
    finally:
        cursor.close()
        connection.close()


@app.route('/')
def index():
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, title, content FROM news ORDER BY id DESC LIMIT 10")
        results = cursor.fetchall()
        return jsonify(results)
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()


# @app.route('/summarize', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     text_to_summarize = data.get('text', '').strip()
#     if not text_to_summarize:
#         return jsonify({'error': 'No text provided to summarize.'}), 400

#     # Generate summaries
#     try:
#         # Generate summary using the generic model
#         generic_response = summarizer(
#             text_to_summarize, max_length=100, min_length=25, do_sample=False)
#         if not generic_response or 'summary_text' not in generic_response[0]:
#             raise ValueError("Generic summary is empty or malformed.")
#         generic_summary = generic_response[0]['summary_text']

#         # Generate summary using the fine-tuned model
#         fine_tuned_response = fine_tuned_model_pipeline(
#             text_to_summarize, max_length=100, min_length=25, do_sample=False)
#         if not fine_tuned_response or 'summary_text' not in fine_tuned_response[0]:
#             raise ValueError("Fine-tuned summary is empty or malformed.")
#         fine_tuned_summary = fine_tuned_response[0]['summary_text']

#     except Exception as e:
#         print(f"Summarization error: {e}")
#         return jsonify({'error': 'Failed to generate summaries'}), 500

#     # Save full content and both summaries to database in one go
#     connection = get_db_connection()
#     if not connection:
#         return jsonify({'error': 'Database connection failed.'}), 500

#     try:
#         cursor = connection.cursor()
#         save_query = """
#         INSERT INTO news_with_summaries (title, content, summarized_content, fine_tuned_summary)
#         VALUES (%s, %s, %s, %s)
#         """
#         title = "Untitled Article"  # Or fetch title from elsewhere if available
#         cursor.execute(save_query, (title, text_to_summarize,
#                        generic_summary, fine_tuned_summary))
#         connection.commit()
#         saved_id = cursor.lastrowid
#     except mysql.connector.Error as e:
#         print(f"Error saving to database: {e}")
#         return jsonify({'error': 'Failed to save data to database'}), 500
#     finally:
#         cursor.close()
#         connection.close()

#     # Generate sentiment plot for both summaries
#     generic_plot = generate_sentiment_plot_and_save_to_csv(
#         generic_summary, generic_summary)
#     fine_tuned_plot = generate_sentiment_plot_and_save_to_csv(
#         fine_tuned_summary, fine_tuned_summary)

#     # Return the response with redirect URL and plot data
#     return jsonify({
#         'redirect_url': url_for('show_summary', summary_id=saved_id),
#         'generic_plot': generic_plot,
#         'fine_tuned_plot': fine_tuned_plot
#     })


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text_to_summarize = data.get('text', '').strip()

    if not text_to_summarize:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Generate summaries
        generic_response = summarizer(
            text_to_summarize, max_length=100, min_length=25)
        generic_summary = generic_response[0]['summary_text']

        fine_tuned_response = fine_tuned_model_pipeline(
            text_to_summarize, max_length=100, min_length=25)
        fine_tuned_summary = fine_tuned_response[0]['summary_text']

        # Generate sentiment plots
        generic_plot = generate_sentiment_plot_and_save_to_csv(
            generic_summary, generic_summary)
        fine_tuned_plot = generate_sentiment_plot_and_save_to_csv(
            fine_tuned_summary, fine_tuned_summary)

        return jsonify({
            'generic_summary': generic_summary,
            'fine_tuned_summary': fine_tuned_summary,
            'generic_plot': generic_plot,
            'fine_tuned_plot': fine_tuned_plot
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 50


@app.route('/summary/<int:summary_id>')
def show_summary(summary_id):
    connection = get_db_connection()
    if not connection:
        return "Failed to connect to the database.", 500

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT content, summarized_content, fine_tuned_summary FROM news_with_summaries WHERE id = %s"
        cursor.execute(query, (summary_id,))
        result = cursor.fetchone()
        if not result:
            return "Summary not found.", 404

            # Check if the required content is available
        if not result['summarized_content'] or not result['fine_tuned_summary']:
            return "No summary available for this ID.", 404

        generic_plot = generate_sentiment_plot_and_save_to_csv(
            result['summarized_content'], result['summarized_content'])
        fine_tuned_plot = generate_sentiment_plot_and_save_to_csv(
            result['fine_tuned_summary'], result['fine_tuned_summary'])

        return render_template(
            'summary.html',
            full_text=result['content'],
            generic_summary=result['summarized_content'],
            fine_tuned_summary=result['fine_tuned_summary'],
            generic_plot=generic_plot,
            fine_tuned_plot=fine_tuned_plot
        )
    except mysql.connector.Error as e:
        print(f"Error fetching summary: {e}")
        return "An error occurred while fetching the summary.", 500
    finally:
        cursor.close()
        connection.close()


if __name__ == '__main__':
    app.run(debug=True)
