import os
import sqlite3
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import requests
import openai
from sklearn.cluster import KMeans
import numpy as np

# Flask app initialization
app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database setup
DATABASE = 'journal_entries.db'

# OpenAI API Key (replace with your actual key)
OPENAI_API_KEY = 'your_openai_api_key'

# Helper function to connect to SQLite3 database
def connect_db():
    conn = sqlite3.connect(DATABASE)
    return conn

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the database table for journal entries
def init_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_number TEXT,
            account_name TEXT,
            standard TEXT,
            general_ledger_entry TEXT,
            transaction_date TEXT,
            currency TEXT,
            accounting_entry_from TEXT,
            accounting_entry_to TEXT,
            amount REAL
        )
    ''')
    conn.commit()
    conn.close()

# API route to store a journal entry in the database
@app.route('/add_journal_entry', methods=['POST'])
def add_journal_entry():
    data = request.json

    account_number = data.get('account_number')
    account_name = data.get('account_name')
    standard = data.get('standard')
    general_ledger_entry = data.get('general_ledger_entry')
    transaction_date = data.get('transaction_date')
    currency = data.get('currency')
    accounting_entry_from = data.get('accounting_entry_from')
    accounting_entry_to = data.get('accounting_entry_to')
    amount = data.get('amount')

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO journal_entries (account_number, account_name, standard, general_ledger_entry,
        transaction_date, currency, accounting_entry_from, accounting_entry_to, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (account_number, account_name, standard, general_ledger_entry, transaction_date, currency, accounting_entry_from, accounting_entry_to, amount))
    conn.commit()
    conn.close()

    return jsonify({"message": "Journal entry added successfully"}), 200

# API route to retrieve all journal entries from the database
@app.route('/get_all_journal_entries', methods=['GET'])
def get_all_journal_entries():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM journal_entries')
    entries = cursor.fetchall()
    conn.close()

    journal_entries = []
    for entry in entries:
        journal_entries.append({
            "id": entry[0],
            "account_number": entry[1],
            "account_name": entry[2],
            "standard": entry[3],
            "general_ledger_entry": entry[4],
            "transaction_date": entry[5],
            "currency": entry[6],
            "accounting_entry_from": entry[7],
            "accounting_entry_to": entry[8],
            "amount": entry[9]
        })

    return jsonify(journal_entries), 200

# Clustering algorithm to find similar journal entries
def cluster_journal_entries(entries, target_entry):
    # We will use a simple k-means clustering algorithm based on numeric values.
    # Convert categorical fields into numerical encoding (e.g., account number, amount).
    
    # Prepare data for clustering
    data = []
    for entry in entries:
        data.append([float(entry['amount'])])
    
    # Convert the target entry for comparison
    target_data = np.array([[float(target_entry['amount'])]])

    # Perform clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)

    # Predict which cluster the target entry belongs to
    target_cluster = kmeans.predict(target_data)[0]

    # Find entries in the same cluster
    similar_entries = []
    for i, label in enumerate(kmeans.labels_):
        if label == target_cluster:
            similar_entries.append(entries[i])

    return similar_entries

# API route to find similar journal entries based on input entry
@app.route('/find_similar_entries', methods=['POST'])
def find_similar_entries():
    # The input journal entry for comparison
    target_entry = request.json

    # Fetch all journal entries from the database
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM journal_entries')
    entries = cursor.fetchall()
    conn.close()

    # Format entries into dictionary
    journal_entries = []
    for entry in entries:
        journal_entries.append({
            "id": entry[0],
            "account_number": entry[1],
            "account_name": entry[2],
            "standard": entry[3],
            "general_ledger_entry": entry[4],
            "transaction_date": entry[5],
            "currency": entry[6],
            "accounting_entry_from": entry[7],
            "accounting_entry_to": entry[8],
            "amount": entry[9]
        })

    # Find similar journal entries
    similar_entries = cluster_journal_entries(journal_entries, target_entry)

    return jsonify(similar_entries), 200

# Initialize the database on startup
init_db()

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
