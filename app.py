from flask import Flask, render_template, request, url_for
import os
import uuid
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pyodbc
import json

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Crée les dossiers s'ils n'existent pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Charge les modèles YOLO
model_infected = YOLO('best.pt')
model_defected = YOLO('last.pt')

# Connexion SQL Server (adapter selon ta config)
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=dataset;'
        'Trusted_Connection=yes;'
    )
    return conn

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            filename = str(uuid.uuid4()) + os.path.splitext(uploaded_file.filename)[1]
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(img_path)

            results_inf = model_infected(img_path)[0]
            boxes_inf = results_inf.boxes

            results_def = model_defected(img_path)[0]
            boxes_def = results_def.boxes

            if len(boxes_inf) > 0:
                prediction = "infected"
                result_img = results_inf.plot()
                # Extraire le score de confiance max
                confidence = max(boxes_inf.conf.tolist()) if boxes_inf.conf is not None else 0.0
                # Action selon seuil 0.5
                action = "rejeter" if confidence > 0.5 else "recycler"
            elif len(boxes_def) > 0:
                prediction = "defected"
                result_img = results_def.plot()
                confidence = max(boxes_def.conf.tolist()) if boxes_def.conf is not None else 0.0
                action = "aucune"
            else:
                prediction = "normal"
                img = Image.open(img_path).convert("RGB")
                result_img = img.copy()
                confidence = 0.0
                action = "aucune"

            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            if isinstance(result_img, np.ndarray):
                result_img = Image.fromarray(result_img)

            result_img.save(result_path)

            # Sauvegarde dans la base SQL Server
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO dbo.dataset 
                    (image_name, chemin_image, infection_class, infection_percent, confiance, decision)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, filename, img_path, prediction, confidence, confidence, action)
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print("Erreur insertion SQL:", e)

            return render_template('index.html',
                                   prediction=prediction,
                                   confidence=confidence,
                                   action=action,
                                   uploaded_image=url_for('static', filename='uploads/' + filename),
                                   result_image=url_for('static', filename='results/' + result_filename))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dbo.dataset")
        total_images = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dbo.dataset WHERE infection_class = 'infected'")
        infected_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dbo.dataset WHERE infection_class = 'defected'")
        defected_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dbo.dataset WHERE decision = 'rejeter'")
        rejeter_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dbo.dataset WHERE decision = 'recycler'")
        recycler_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

    except Exception as e:
        print("Erreur récupération SQL:", e)
        total_images = infected_count = defected_count = rejeter_count = recycler_count = 0

    classes = ['infected', 'defected', 'normal']
    counts = [infected_count, defected_count, max(total_images - infected_count - defected_count, 0)]

    actions = ['rejeter', 'recycler', 'aucune']
    actions_counts = [rejeter_count, recycler_count, max(total_images - rejeter_count - recycler_count, 0)]

    return render_template('dashboard.html',
                           total_images=total_images,
                           infected_count=infected_count,
                           defected_count=defected_count,
                           rejeter_count=rejeter_count,
                           recycler_count=recycler_count,
                           classes=json.dumps(classes),
                           counts=json.dumps(counts),
                           actions=json.dumps(actions),
                           actions_counts=json.dumps(actions_counts))

if __name__ == '__main__':
    app.run(debug=True)
