from flask import Flask, render_template, request, session
from boolean_model import BooleanModel 
from dist_boolean_model import DistBooleanModel 
# from sparse import gen_wordcloud, gen_titles, gen_keywords, gen_abstracts


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

model = BooleanModel(content_path="src/Content.csv")  # Initialize your BooleanModel instance
# model = DistBooleanModel(content_path="src/Content.csv")  # Initialize your BooleanModel instance


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    schema = request.form['schema']
    n = int(request.form['ngram'])
    paper_ids, titles, _, _ = model.compute_ranking(query, schema, n=n, threshold=50)
    
    # Store search results in session
    session['paper_ids'] = ";".join(paper_ids)
    
    return render_template('search_results.html', titles=titles)


@app.route('/details/<int:index>')
def details(index):
    # Retrieve search results from session
    paper_ids = session.get('paper_ids').split(";")

    # Retrieve details of the selected paper
    filtered_df = model.df_content[model.df_content['paper_id'].isin(paper_ids)]
    title = filtered_df['title'].tolist()[index]
    keywords = filtered_df['keywords'].tolist()[index]
    abstract = filtered_df['abstract'].tolist()[index]

    return render_template('details.html', title=title, keywords=keywords, abstract=abstract)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True, use_reloader=False)  # disable reloader if you face issues with Spark contexts