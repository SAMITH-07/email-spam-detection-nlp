# Email Spam Detection using NLP

A machine learning project that classifies emails as spam or ham (not spam) using Natural Language Processing and Multinomial Naive Bayes algorithm with a Flask web interface.

## 🎯 Features

- **Real-time Email Classification**: Instantly classify emails as spam or ham
- **NLP Processing**: Advanced text preprocessing with stemming and stopword removal
- **Web Interface**: User-friendly Flask web application
- **High Accuracy**: Multinomial Naive Bayes classifier with good performance
- **Visual Feedback**: Animated spam/ham indicators
- **Pre-trained Model**: Ready-to-use trained model and vectorizer

## 📁 Project Structure

```
email-spam-detection-nlp/
├── app.py                    # Flask web application
├── spam.py                   # Model training script
├── model.pkl                 # Trained Naive Bayes model
├── cv-transform.pkl          # CountVectorizer transformer
├── EmailCollection.csv       # Dataset for training
├── templates/                # HTML templates
│   ├── home.html            # Home page with input form
│   └── result.html          # Results display page
├── static/                   # Static assets
│   ├── styles.css           # Styling
│   ├── spam.gif             # Spam animation
│   ├── not-spam.gif         # Ham animation
│   └── *.webp, *.ico        # Additional assets
└── README.md                 # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies
```bash
pip install flask pandas numpy scikit-learn nltk seaborn matplotlib
```

### Download NLTK Data
The training script will automatically download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🎮 Usage

### Option 1: Use Pre-trained Model (Recommended)
1. **Run the Web Application**:
   ```bash
   python app.py
   ```
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Test Email Classification**: Enter email text and click "Predict"

### Option 2: Train Your Own Model
1. **Prepare Dataset**: Place your email dataset as `EmailCollection.csv`
   - Format: Tab-separated with columns `LABEL` and `MESSAGES`
   - Labels: 'ham' for legitimate emails, 'spam' for spam emails

2. **Train the Model**:
   ```bash
   python spam.py
   ```
   - This will train the model and save `model.pkl` and `cv-transform.pkl`

3. **Run Web Application**:
   ```bash
   python app.py
   ```

## 🧠 How It Works

### 1. Data Preprocessing
- **Text Cleaning**: Removes non-alphabetic characters
- **Lowercase Conversion**: Converts text to lowercase
- **Tokenization**: Splits text into individual words
- **Stopword Removal**: Removes common English stopwords
- **Stemming**: Reduces words to their root form using Porter Stemmer

### 2. Feature Extraction
- **CountVectorizer**: Converts text to numerical features
- **Max Features**: Uses top 3500 most frequent words
- **Bag of Words**: Creates word frequency vectors

### 3. Classification
- **Algorithm**: Multinomial Naive Bayes
- **Alpha Parameter**: 0.8 for smoothing
- **Training**: 80% train, 20% test split
- **Performance**: High accuracy on email classification

### 4. Web Interface
- **Flask Framework**: Lightweight web server
- **User Input**: HTML form for email text
- **Real-time Prediction**: Instant classification results
- **Visual Feedback**: Animated spam/ham indicators

## 📊 Model Performance

The Multinomial Naive Bayes classifier achieves high accuracy on email spam detection:
- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: CountVectorizer (max_features=3500)
- **Text Processing**: Stemming + Stopword Removal
- **Test Split**: 20% of dataset
- **Expected Accuracy**: >95% (varies by dataset)

## 🌐 Web Interface Features

### Home Page (`/`)
- Clean, responsive design
- Text area for email input
- Submit button for classification
- Professional styling with CSS

### Results Page (`/predict`)
- Displays prediction result
- Visual spam/ham indicators
- Option to classify another email
- Clear result presentation

## 📧 Example Usage

### Spam Email Example
```
WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
```
**Result**: Spam ✗

### Ham Email Example
```
Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
```
**Result**: Ham ✓

## 🔧 Technical Details

### Dependencies
- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **nltk**: Natural language processing
- **numpy**: Numerical operations
- **seaborn/matplotlib**: Data visualization

### Model Files
- `model.pkl`: Serialized Multinomial Naive Bayes model
- `cv-transform.pkl`: Serialized CountVectorizer transformer
- Both files are automatically generated during training

## 🛠️ Customization

### Modify Model Parameters
Edit `spam.py` to adjust:
- `max_features` in CountVectorizer
- `alpha` parameter in Multinomial Naive Bayes
- `test_size` in train_test_split

### Add New Features
- **TF-IDF Vectorizer**: Replace CountVectorizer for better performance
- **Additional Algorithms**: Try SVM, Random Forest, or Neural Networks
- **Email Features**: Add sender, subject, timestamp features
- **Deep Learning**: Implement LSTM or BERT models

## 🐛 Troubleshooting

### Common Issues
1. **Module Not Found**: Install all required dependencies
2. **NLTK Data Missing**: Run NLTK downloads manually
3. **Model Loading Error**: Ensure `model.pkl` and `cv-transform.pkl` exist
4. **Port Already in Use**: Change Flask port or stop conflicting services

### Performance Issues
- **Large Dataset**: Reduce `max_features` or use streaming
- **Memory Issues**: Process data in batches
- **Slow Prediction**: Optimize vectorizer or use lighter models

## 📈 Future Enhancements

- **Real-time Email Integration**: Gmail/Outlook API integration
- **Multi-language Support**: Support for emails in different languages
- **Advanced Features**: Email header analysis, sender reputation
- **Deep Learning**: BERT or transformer-based models
- **API Development**: RESTful API for integration
- **Mobile App**: React Native or Flutter application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**SAMITH-07**
- GitHub: [SAMITH-07](https://github.com/SAMITH-07)

## 🙏 Acknowledgments

- scikit-learn for machine learning tools
- NLTK for natural language processing
- Flask for web framework
- Email spam detection research community

---

**Note**: This project is for educational purposes. For production email filtering, consider using established email security solutions with additional features like sender reputation, IP blacklisting, and real-time threat intelligence.
