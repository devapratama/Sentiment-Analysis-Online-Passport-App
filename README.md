# Sentiment Analysis for Indonesian Immigration Online Passport Application

## Project Overview

This project analyzes user sentiment regarding the Indonesian Immigration Online Passport Application by leveraging reviews scraped from the Google Play Store. The objective is to classify user feedback into positive, neutral, and negative sentiments, providing actionable insights to improve the application’s user experience and functionality. The analysis employs machine learning models (Random Forest, Support Vector Machine, XGBoost) and deep learning (BERT-based transformer models) to achieve high accuracy and reliability in sentiment classification.

---

## Motivation

The Indonesian Immigration Online Passport Application has received significant user feedback, with common issues such as application crashes, slow verification, and unhelpful support systems frequently mentioned. Analyzing this feedback provides a pathway for:
- Identifying key pain points affecting user satisfaction.
- Highlighting areas for immediate improvement.
- Delivering actionable recommendations to enhance overall user experience.

This project is inspired by the critical role of user sentiment in driving application improvements, particularly in services that affect millions of users.

---

## Business Questions

1. What are the major factors influencing user sentiment regarding the Indonesian Immigration Online Passport Application?
<!--2. How do positive, neutral, and negative sentiments distribute across user reviews?-->
2. Can machine learning and deep learning approaches provide an effective solution for accurate sentiment classification?

---

## Dataset Description

The dataset consists of user reviews scraped from the Google Play Store using the `google_play_scraper` library. The data includes:

| **Column** | **Description** |
|------------|-----------------|
| `Review`   | User's feedback text about the application. |
| `Rating`   | Numeric rating given by the user (1-5). |
| `Date`     | Date and time when the review was submitted. |

### Dataset Statistics
- **Total Records**: 14,342 reviews.
- **Data Range**: Contains reviews from 2024-10-24 and earlier.
- **Key Characteristics**: No missing values; balanced distribution of review lengths.

---

## Methodology

### Data Preprocessing

1. **Text Cleaning**:
   - Removed punctuation, numbers, and excessive whitespace.
   - Translated non-English text into English using `deep_translator` for consistency.
   - Tokenized text into words using NLTK and SpaCy.

2. **Feature Engineering**:
   - **TF-IDF**: Extracted term frequency-inverse document frequency features.
   - **Word2Vec**: Generated word embeddings using Gensim.

3. **Label Encoding**:
   - Mapped sentiment labels as follows: ‘Positive’ -> 2, ‘Neutral’ -> 1, ‘Negative’ -> 0.

### Model Development

#### Machine Learning Models:
1. **Random Forest Classifier**
2. **Support Vector Machine (SVM)**
3. **XGBoost Classifier**

#### Deep Learning Model:
- Fine-tuned a pre-trained `BERT-base-uncased` model for multi-class sentiment classification using PyTorch and Hugging Face’s `transformers` library.

### Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---

## Key Metrics and Results

### Deep Learning (BERT-based Transformer)
| Metric      | Value |
|-------------|-------|
| **Accuracy**| 0.9833|
| **Precision**| 0.9835|
| **Recall**  | 0.9833|
| **F1-Score**| 0.9833|

### Best Performing Machine Learning Model (Random Forest with TF-IDF)
| Metric      | Value |
|-------------|-------|
| **Accuracy**| 0.95|
| **Precision** (macro avg) | 0.95|
| **Recall** (macro avg) | 0.95|
| **F1-Score** (macro avg) | 0.95|

<!--


Visualization
Sentiment Distribution Bar Chart: A bar chart was created to visualize the distribution of sentiments from over 14,000 user reviews of the Indonesian Immigration Online Passport Application. The chart shows the count of reviews categorized as Neutral, Positive, and Negative, providing a clear overview of user sentiment distribution.
Word Cloud: A word cloud was generated from the translated reviews to highlight the most frequently mentioned words, giving insight into common themes and issues discussed by users. Key terms like "application," "difficult," "time," "quota," "error," and "fix" were prominent, indicating user focus and pain points.

Results and Insights
Sentiment Distribution: The analysis revealed that the majority of reviews (6,321) were Neutral, suggesting a significant portion of users had a lukewarm or indifferent experience. Negative sentiments were substantial with 4,966 reviews, while Positive reviews were the least common at 2,677. This indicates a need for addressing user dissatisfaction.
Word Cloud Insights: The word cloud emphasized user frustrations with the application process, highlighting issues like difficulty in use, time-related complaints, technical errors, and quota restrictions. Positive aspects were also noted, with words like "easy" and "good" appearing, though less frequently.

Conclusion and Recommendations
Key Findings:
The majority of user feedback on the application is neutral, with a significant portion expressing negative sentiments, pointing to areas needing improvement in user experience.
Common themes in user feedback include complexity of the application process, long processing times, technical glitches, and quota limitations, with some acknowledgment of positive aspects.

Recommendations:
Simplify User Interface and Process: Enhance the application's user interface to make the process more intuitive and less complicated, reducing user frustration as highlighted by the word "difficult" in the word cloud.
Improve Technical Stability: Address the technical issues like errors and bugs, as seen with words like "error" and "fix" in the word cloud, to ensure a smoother user experience.
Optimize Processing Times: Focus on reducing wait times and speeding up the application process, which was a significant concern reflected by the prominence of "time" in the word cloud.
Increase Quota Availability: Review and possibly increase the availability of quotas or provide clearer information on quota management, as "quota" was a notable concern in user reviews.

These insights and recommendations are derived from the sentiment distribution and word cloud analysis, aiming to enhance user satisfaction with the Indonesian Immigration Online Passport Application. -->

---

## Conclusion and Recommendations

### Key Findings:
1. **Critical User Issues**: Negative sentiments dominate due to technical bugs and inadequate customer service.
2. **Model Performance**: Deep learning (BERT) provides superior accuracy, making it a reliable choice for future sentiment analysis tasks.

### Recommendations:
1. Address critical user complaints by:
   - Improving application stability and performance.
   - Streamlining the verification process.
   - Enhancing customer support response times.
2. Regularly monitor user sentiment using the developed deep learning model to gauge improvements post-implementation.

---

## Next Steps
1. Incorporate additional external data sources (e.g., social media comments, app crash reports).
2. Perform time-series analysis on sentiment trends to identify the impact of updates or changes to the application.
3. Explore multi-lingual sentiment analysis to cater to non-Indonesian users.

---

## Tools and Skills Used
- **Tools**: Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Hugging Face Transformers, PyTorch), Gensim, Google Play Scraper.
- **Skills**: Data scraping, text preprocessing, feature engineering, machine learning, deep learning, model evaluation, visualization.

---

## References
1. Google Play Scraper: [https://github.com/JoMingyu/google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
2. BERT Pre-trained Models: [https://github.com/google-research/bert](https://github.com/google-research/bert)
