import re
import subprocess
from flask import Flask, render_template, request, session
from boolean_model import BooleanModel 


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management


# Initialize the Normal BooleanModel instance.
model = BooleanModel(content_path="/home/zhengli/Documents/CSCI-626/project/src/Content.csv")
# Condiguration of the distributed Boolean Model.
python_interpreter = "/home/zhengli/anaconda3/envs/nlpenv/bin/python"
dist_scripts = "/home/zhengli/Documents/CSCI-626/project/dist_boolean_model.py"
content_path = "/home/zhengli/Documents/CSCI-626/project/src/Content.csv"



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    schema = request.form['schema']
    n = int(request.form['ngram'])

    if request.form['model'] == "dist_boolean_model":
        try:
            # Call the distributed Boolean Model
            command = [
                python_interpreter, dist_scripts,
                "--path", content_path, 
                "--query", query, 
                "--schema", schema, 
                "--n", str(n)
            ]
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Regular expression to parse the data
            pattern = r"\[([^\]]+)\]\n([^\n]+)\n\[(NULL)\]"

            # Find all matches
            matches = re.findall(pattern, result.stdout)

            # Extract paper_ids, titles, and keywords
            paper_ids = [match[0] for match in matches]
            titles = [match[1] for match in matches]
            keywords = [match[2] for match in matches]
        except subprocess.CalledProcessError as e:
            print("Error during script execution:", e.stderr)
        except PermissionError as e:
            print("Permission error:", e)
    else:
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
    app.run(debug=True)
    # app.run(debug=True, use_reloader=False)  # disable reloader if you face issues with Spark contexts


