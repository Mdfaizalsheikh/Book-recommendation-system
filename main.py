import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import messagebox

class BookRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Recommendation System")

        
        self.df = self.load_data('books.txt')

        self.df['combined_features'] = self.df['title'] + ' ' + self.df['author'] + ' ' + self.df['genres'] + ' ' + self.df['description']

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])

        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        
        self.label = tk.Label(self.root, text="Enter a Book Title:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)

        self.button = tk.Button(self.root, text="Get Recommendations", command=self.get_recommendations)
        self.button.pack(pady=10)

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                title, author, genres, description = line.strip().split('|')
                data.append({'title': title, 'author': author, 'genres': genres, 'description': description})
        return pd.DataFrame(data)

    def get_recommendations(self):
        title = self.entry.get().strip()
        if title:
            try:
                recommendations = self.recommend_books(title)
                if recommendations:
                    messagebox.showinfo("Recommendations", "\n".join(recommendations))
                else:
                    messagebox.showwarning("No Recommendations", "No recommendations found for this title.")
            except IndexError:
                messagebox.showerror("Error", "Book title not found.")
        else:
            messagebox.showerror("Error", "Please enter a book title.")

    def recommend_books(self, title, num_recommendations=5):
        idx = self.df[self.df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        book_indices = [i[0] for i in sim_scores]
        return self.df['title'].iloc[book_indices].tolist()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = BookRecommendationApp(root)
    app.run()
