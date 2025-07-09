# Automated Candidate Selection System for Recruitment Using NLP

This project is an **Automated Candidate Selection System** that leverages Natural Language Processing (NLP) and Machine Learning to parse, analyze, and classify resumes. It helps recruitment teams efficiently match candidates to job roles based on their skills, experience, and education.

---

## Features

- **Resume Parsing:** Extracts structured information (skills, education, experience, contact info) from PDF resumes.
- **NLP-based Classification:** Classifies resumes into relevant job categories using machine learning models.
- **Skill & Experience Extraction:** Identifies and quantifies candidate skills and work experience for ranking.
- **Automated Matching:** Matches candidates to roles based on extracted features and a labeled dataset.
- **Visualization:** Generates confusion matrices and word clouds for model and content analysis.
- **Extensible Pipeline:** Easily add new features, models, or data sources.

---

## Project Structure


| File/Folder                  | Description                                         |
|------------------------------|-----------------------------------------------------|
| `resume/`                    | Folder containing PDF resumes                       |
| `resume_dataset.csv `        | Labeled dataset for training and evaluation         |
| `DataInitialise.py `         | Data loading and preprocessing scripts              |
| `InfoExtractor.py`           | Information extraction logic from resumes           |
| `NLP-Classifier.py`          | NLP model training and evaluation                   |
| `Resume_classifier.py`       | Additional classifier logic                         |
| `Resume_classifier.py`       | Additional classifier logic                         |
| `ResumeHelper.py  `          | Helper functions for text extraction                |
| `ResumeSegmenter.py `        | Resume segmentation and normalization               |
| `SegementerClassifier.pyy`   | Feature extraction and segmentation classifiers     |
| `TrainingModel.py `          | Model training and evaluation utilities             |
| `utils.py `                  | Utility functions                                   |
| `README.md   `               | Project documentation                               |

---

## Installation

1. **Clone the repository:**

```
git clone https://github.com/yourusername/automated-candidate-selection-nlp.git
cd automated-candidate-selection-nlp
```

2. **Download NLTK data (if required):**

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

1. **Prepare Data:**
- Place all candidate resumes (PDF format) in the `resume/` folder.
- Ensure `resume_dataset.csv` contains labeled resume data for training.

2. **Train the Model:**
- Run the classifier script to train models:
  ```
  python NLP-Classifier.py
  ```
- Or use `TrainingModel.py` for advanced training options.

3. **Extract Information from New Resumes:**
- Use `InfoExtractor.py` to parse and extract candidate details:
  ```
  python InfoExtractor.py --resume resume/sample_resume.pdf
  ```

4. **Classify and Match Candidates:**
- The system will output candidate details, skillset, experience, and predicted job category.

---

## Technologies Used

- **Programming Languages:** Python (pandas, numpy, scipy, scikit-learn, matplotlib), SQL, Java, JavaScript/jQuery
- **Machine Learning:** Regression, SVM, Naive Bayes, KNN, Random Forest, Decision Trees, Boosting, Clustering, Topic Modelling (LDA, NMF), PCA, Neural Nets
- **Visualization:** matplotlib, seaborn, wordcloud, Plotly, Tableau
- **NLP:** tfidf, word2vec, doc2vec, cosine similarity, Vader, TextBlob
- **Others:** HTML, CSS, Angular, Flask, Docker, Git, OpenCV

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements
Inspired by real-world recruitment needs and powered by open-source NLP and ML libraries.
