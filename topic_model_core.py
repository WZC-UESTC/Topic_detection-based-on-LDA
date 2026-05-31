import re
import string

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


FALLBACK_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this",
    "that", "from", "have", "has", "had", "was", "were", "will", "would",
    "can", "could", "should", "about", "into", "than", "then", "there",
    "their", "them", "they", "its", "our", "out", "who", "what", "when",
    "where", "which", "why", "how", "all", "any", "each", "more", "most",
    "other", "some", "such", "only", "own", "same", "too", "very", "also",
    "been", "being", "over", "under", "after", "before", "between"
}


class TopicModelCore:
    """Topic detection core shared by the web wrapper."""

    def __init__(self):
        self.documents = []
        self.doc_names = []
        self.processed_docs = []
        self.vectorizer = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.feature_names = None
        self.topics = None

    def load_documents(self, docs, names=None):
        self.documents = docs
        self.doc_names = names if names else [f"doc_{i + 1}.txt" for i in range(len(docs))]

    def preprocess_documents(self):
        self.processed_docs = []

        for doc in self.documents:
            text = doc.lower()
            text = re.sub(r"\d+", "", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            tokens = re.findall(r"[a-zA-Z][a-zA-Z_'-]*|[\u4e00-\u9fff]+", text)
            filtered_tokens = [
                token for token in tokens
                if len(token) > 2 and token not in FALLBACK_STOP_WORDS
            ]
            self.processed_docs.append(" ".join(filtered_tokens))

        return self.processed_docs

    def vectorize_documents(self, min_df=None, max_df=None, max_features=600):
        num_docs = len(self.processed_docs)
        min_df = min_df if min_df is not None else (1 if num_docs < 3 else 2)
        max_df = max_df if max_df is not None else (1.0 if num_docs < 3 else 0.85)

        self.vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=min_df,
            stop_words="english",
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b",
        )

        try:
            self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
        except ValueError as exc:
            if "After pruning, no terms remain" not in str(exc):
                raise
            self.vectorizer = CountVectorizer(
                max_df=1.0,
                min_df=1,
                stop_words="english",
                max_features=max_features,
                token_pattern=r"(?u)\b\w+\b",
            )
            self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)

        if hasattr(self.vectorizer, "get_feature_names_out"):
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            self.feature_names = np.array(self.vectorizer.get_feature_names())
        return self.doc_term_matrix

    def train_lda(self, num_topics=5, max_iter=12):
        n_topics = max(1, min(num_topics, len(self.processed_docs)))
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="online",
            random_state=42,
            n_jobs=1,
            batch_size=64,
        )
        self.lda_model.fit(self.doc_term_matrix)
        return self.lda_model

    def extract_topics(self, num_words=10):
        num_words = min(num_words, len(self.feature_names))
        self.topics = []

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-num_words - 1:-1]
            self.topics.append({
                "index": topic_idx,
                "words": [self.feature_names[i] for i in top_indices],
                "weights": [topic[i] for i in top_indices],
            })

        return self.topics

    def get_doc_topic_distribution(self):
        return self.lda_model.transform(self.doc_term_matrix)

    def run_pipeline(
        self,
        docs,
        num_topics=5,
        num_words=10,
        max_iter=12,
        min_df=None,
        max_df=None,
        max_features=600,
    ):
        self.load_documents(docs)
        self.preprocess_documents()
        self.vectorize_documents(min_df=min_df, max_df=max_df, max_features=max_features)
        self.train_lda(num_topics, max_iter)
        self.extract_topics(num_words)
        return self.topics, self.get_doc_topic_distribution()
