# Blood Donor Eligibility Classification System

A web-based application that classifies blood donor eligibility using the **Naïve Bayes machine learning algorithm**. The system processes donor information, physical examination results, and medical history to predict whether a potential donor is **Eligible** or **Not Eligible**.

<escape> </escape>

## 🌐 Live Demo

**[Access Live Demo](https://hikamsrl.pythonanywhere.com/)**

## ✨ Features

- **User Authentication** — Secure login and registration for application users.
- **Donor Information Management** — Input and manage donor personal information.
- **Physical Examination** — Record age, body weight, hemoglobin, blood pressure, pulse rate, and body temperature.
- **Medical History** — Record medical conditions and relevant health information.
- **Naïve Bayes Classification** — Predict donor eligibility using a trained machine learning model.
- **Eligibility Validation** — Apply predefined screening criteria to support the classification process.
- **Classification History** — Store and review previous classification results.
- **Model Evaluation** — Display machine learning model evaluation information.

## 🧠 Machine Learning

This application uses the **Naïve Bayes algorithm**, a probabilistic classification method based on Bayes' Theorem.

The classification process involves:

1. Data Collection
2. Data Preprocessing
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Deployment

The trained machine learning model is stored in `.pkl` format using Joblib and loaded by the Flask application for prediction.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| Scikit-learn | Machine learning implementation |
| Pandas | Data processing and preprocessing |
| NumPy | Numerical computation |
| Joblib | Model serialization and loading |
| MySQL | Relational database |
| Flask-SQLAlchemy | Database ORM |
| Flask-Login | User authentication and session management |
| HTML | Web page structure |
| Tailwind CSS | User interface styling |
| JavaScript | Frontend interaction and API communication |
| Chart.js | Data visualization |
| DataTables | Interactive data tables |

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HikamSirrul/Klasifikasi-Donor-Naive-Bayes.git
```

Navigate to the project directory:

```bash
cd Klasifikasi-Donor-Naive-Bayes
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Database

Create a MySQL database:

```sql
CREATE DATABASE skripsi_pmi;
```

Configure the database connection settings in the application according to your local MySQL credentials.

### 5. Configure the Machine Learning Model

Make sure the trained model file is available in the expected project directory:

```text
model_naive_bayes_pmi.pkl
```

The file contains the trained Naïve Bayes model and encoder required for data preprocessing and classification.

### 6. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 📊 Classification Results

The application provides two classification outcomes:

| Result | Description |
|---|---|
| Eligible | The system classifies the potential donor as meeting the implemented eligibility criteria. |
| Not Eligible | The system classifies the potential donor as not meeting the implemented eligibility criteria. |

**Disclaimer:** Classification results are intended as decision-support information and should not replace professional medical screening or decisions made by authorized medical personnel.

## 🔒 Security and Privacy

- Protect personal and medical information.
- Keep database credentials confidential.
- Restrict access to authorized users.
- Do not upload real donor records or sensitive information to a public repository.

## 👨‍💻 Developer

**Hikam Sirrul Arifin**

Informatics Engineering Graduate | Full-Stack Web Developer

**Areas of Interest:**
- Software Development
- Full-Stack Web Development
- Backend Development
- Machine Learning
- Cybersecurity

## 📄 License

This project is developed for educational and portfolio purposes. Further use, modification, and distribution are subject to the license specified by the repository owner.

---

Developed by **Hikam Sirrul Arifin**.
