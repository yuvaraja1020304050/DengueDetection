from flask import Flask, render_template, request, redirect, url_for, flash, session
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import io
import base64
import struct
from sklearn.metrics import accuracy_score
from contextlib import contextmanager

app = Flask(__name__)
app.secret_key = 'dengue_prediction_secret_key'

# Load trained models
def load_models():
    models = {}
    model_names = ['random_forest_model', 'xgboost_model', 'svm_model']
    for model_name in model_names:
        model_path = os.path.join('models', f'{model_name}.pkl')
        if os.path.exists(model_path):
            models[model_name] = pickle.load(open(model_path, 'rb'))
    
    # Load feature names
    feature_names_path = os.path.join('models', 'feature_names.pkl')
    if os.path.exists(feature_names_path):
        models['feature_names'] = pickle.load(open(feature_names_path, 'rb'))
    
    return models

models = load_models()

# Improved database connection handling
def get_db_connection():
    conn = sqlite3.connect('instance/dengue_prediction.db', timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging for better concurrency
    conn.execute('PRAGMA busy_timeout = 5000')  # Set busy timeout to 5 seconds
    return conn

@contextmanager
def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            nsi REAL NOT NULL,
            igg REAL NOT NULL,
            area TEXT NOT NULL,
            area_type TEXT NOT NULL,
            house_type TEXT NOT NULL,
            district TEXT NOT NULL,
            outcome INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_date TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()

# Initialize database
init_db()

# Helper function to safely extract confidence value
def extract_confidence(prediction):
    confidence_value = prediction['confidence']
    
    # If it's already a number, return it
    if isinstance(confidence_value, (int, float)):
        return confidence_value
    
    # If it's bytes, try different approaches
    if isinstance(confidence_value, bytes):
        try:
            # Try to decode as string and convert to float
            return float(confidence_value.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            try:
                # Try to interpret as a 32-bit float (common binary format)
                if len(confidence_value) == 4:
                    return struct.unpack('f', confidence_value)[0]
                elif len(confidence_value) == 8:
                    return struct.unpack('d', confidence_value)[0]
            except struct.error:
                pass
    
    # Default to a middle value if we can't determine
    return 50.0

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        
        try:
            with get_db() as conn:
                # Check if username exists
                user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
                
                if user:
                    flash('Username already exists!')
                    return redirect(url_for('register'))
                
                # Check if email exists
                email_check = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
                if email_check:
                    flash('Email already registered!')
                    return redirect(url_for('register'))
                
                hashed_password = generate_password_hash(password)
                
                conn.execute(
                    'INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
                    (username, hashed_password, name, email)
                )
                conn.commit()
                
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
        except sqlite3.Error as e:
            flash(f'Database error: {e}')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            with get_db() as conn:
                user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    flash('Login successful!')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password')
        except sqlite3.Error as e:
            flash(f'Database error: {e}')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    try:
        with get_db() as conn:
            predictions_count = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', 
                                            (session['user_id'],)).fetchone()[0]
            positive_count = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ? AND outcome = 1', 
                                        (session['user_id'],)).fetchone()[0]
            last_predictions_raw = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT 5', 
                                            (session['user_id'],)).fetchall()
        
        # Process the last predictions to handle the confidence value safely
        last_predictions = []
        for prediction in last_predictions_raw:
            pred_dict = dict(prediction)
            pred_dict['confidence'] = extract_confidence(prediction)
            last_predictions.append(pred_dict)
        
        return render_template('dashboard.html', 
                            predictions_count=predictions_count,
                            positive_count=positive_count,
                            last_predictions=last_predictions)
    except sqlite3.Error as e:
        flash(f'Database error: {e}')
        return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        gender = request.form['gender']
        age = int(request.form['age'])
        nsi = float(request.form['nsi'])
        igg = float(request.form['igg'])
        area = request.form['area']
        area_type = request.form['area_type']
        house_type = request.form['house_type']
        district = request.form['district']
        model_choice = request.form['model']
        
        input_data = []
        
        # One-hot encoding
        input_data.extend([1, 0] if gender == 'male' else [0, 1])
        input_data.append(age)
        input_data.append(nsi)
        input_data.append(igg)
        
        areas = ['urban', 'rural', 'suburban']
        input_data.extend([1 if a == area else 0 for a in areas])
        
        area_types = ['residential', 'commercial', 'industrial']
        input_data.extend([1 if at == area_type else 0 for at in area_types])
        
        house_types = ['apartment', 'house', 'slum']
        input_data.extend([1 if ht == house_type else 0 for ht in house_types])
        
        districts = ['district1', 'district2', 'district3', 'district4', 'district5']
        input_data.extend([1 if d == district else 0 for d in districts])
        
        input_array = np.array(input_data).reshape(1, -1)
        
        if model_choice == 'random_forest':
            model_key = 'random_forest_model'
            model_name = 'Random Forest'
        elif model_choice == 'xgboost':
            model_key = 'xgboost_model'
            model_name = 'XGBoost'
        else:
            model_key = 'svm_model'
            model_name = 'SVM'
        
        prediction = models[model_key].predict(input_array)[0]
        
        if hasattr(models[model_key], 'predict_proba'):
            confidence = models[model_key].predict_proba(input_array)[0][1] * 100
        else:
            decision = models[model_key].decision_function(input_array)[0]
            confidence = (1 / (1 + np.exp(-decision))) * 100
        
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute('''
                INSERT INTO predictions 
                (user_id, gender, age, nsi, igg, area, area_type, house_type, district, outcome, model_used, confidence, prediction_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session['user_id'], gender, age, nsi, igg, area, area_type, house_type, district, 
                    int(prediction), model_name, confidence, datetime.now()
                ))
                conn.commit()
                prediction_id = cur.lastrowid
            
            try:
                dataset = pd.read_csv('dengue_dataset.csv')
                new_data = pd.DataFrame({
                    'gender': [gender],
                    'age': [age],
                    'nsi': [nsi],
                    'igg': [igg],
                    'area': [area],
                    'area_type': [area_type],
                    'house_type': [house_type],
                    'district': [district],
                    'outcome': [int(prediction)]
                })
                dataset = pd.concat([dataset, new_data], ignore_index=True)
                dataset.to_csv('dengue_dataset.csv', index=False)
            except Exception as e:
                print(f"Error updating dataset: {e}")
            
            return redirect(url_for('result', prediction_id=prediction_id))
        except sqlite3.Error as e:
            flash(f'Database error: {e}')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    try:
        with get_db() as conn:
            prediction_row = conn.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', 
                                      (prediction_id, session['user_id'])).fetchone()
        
        if not prediction_row:
            flash('Prediction not found')
            return redirect(url_for('dashboard'))
        
        # Convert the Row object to a dictionary
        prediction = dict(prediction_row)
        
        # Use the extract_confidence function to get a proper floating-point value
        prediction['confidence'] = extract_confidence(prediction_row)
        
        result_text = "Positive" if prediction['outcome'] == 1 else "Negative"
        risk_level = "High" if prediction['confidence'] > 75 else "Medium" if prediction['confidence'] > 50 else "Low"
        
        return render_template('result.html', prediction=prediction, result_text=result_text, risk_level=risk_level)
    except sqlite3.Error as e:
        flash(f'Database error: {e}')
        return redirect(url_for('dashboard'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    try:
        with get_db() as conn:
            predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC', 
                                    (session['user_id'],)).fetchall()
        
        # Also modify the history page to handle the confidence value safely
        processed_predictions = []
        for prediction in predictions:
            pred_dict = dict(prediction)
            pred_dict['confidence'] = extract_confidence(prediction)
            processed_predictions.append(pred_dict)
        
        return render_template('history.html', predictions=processed_predictions)
    except sqlite3.Error as e:
        flash(f'Database error: {e}')
        return redirect(url_for('dashboard'))

@app.route('/compare_models')
def compare_models():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    try:
        with open('static/model_comparison_report.md', 'r') as f:
            comparison_report = f.read()
    except:
        comparison_report = "Model comparison report not available."
    
    models_accuracy = {
        'Random Forest': 92.5,
        'XGBoost': 91.8,
        'SVM': 88.7
    }
    
    try:
        labels = list(models_accuracy.keys())
        accuracies = list(models_accuracy.values())
        
        fig, ax = plt.subplots()
        ax.bar(labels, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        plt.ylim(0, 100)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        return render_template('compare_models.html', comparison_report=comparison_report, plot_url=plot_url)
    except Exception as e:
        flash(f'Error generating plot: {e}')
        return render_template('compare_models.html', comparison_report=comparison_report, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)