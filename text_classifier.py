import numpy as np
import warnings
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

class NaiveBayesClassifier:
    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        predictions = []
        
        for document in X:
            class_likelihoods = {}
            for cls in classes:
                class_likelihoods[cls] = np.log(class_probs[cls])

                for idx in range(len(document)):
                    if document[idx] > 0:  
                        class_likelihoods[cls] += np.log(word_probs[cls].get(idx, 1e-6))

            predicted_class = max(class_likelihoods, key=class_likelihoods.get)
            predictions.append(predicted_class)

        return predictions
    
    @staticmethod
    def preprocess(sentences, categories):       
        cleaned_sentences = []
        valid_categories = []

        for i in range(len(sentences)):
            sentence = sentences[i]
            category = categories[i]

            if category is None or category == "wrong_label":
                continue
            
            words = sentence.split()
            filtered_sentence = ' '.join([word for word in words if word.lower() not in ENGLISH_STOP_WORDS])

            cleaned_sentences.append(filtered_sentence)
            valid_categories.append(category)

        return cleaned_sentences, valid_categories 
        
    @staticmethod
    def fit(X, y):
        class_probs = {}
        word_conditional_probs = defaultdict(lambda: defaultdict(lambda: 0))

        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_docs = len(y)

        for i in range(len(unique_classes)):
            cls = unique_classes[i]
            count = class_counts[i]
            class_probs[cls] = count / total_docs

        for cls in unique_classes:
            class_indices = [i for i in range(len(y)) if y[i] == cls]
            word_count_per_class = np.sum(X[class_indices], axis=0)
            total_words = np.sum(word_count_per_class)
            
            for idx in range(len(word_count_per_class)):
                word_conditional_probs[cls][idx] = (word_count_per_class[idx] + 1) / (total_words + len(word_count_per_class))

        return class_probs, word_conditional_probs

