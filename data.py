from flask import Flask, request, send_file
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')

        output = io.BytesIO()
        i=0
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for idx, table in enumerate(tables):
                df = pd.read_html(str(table))[0]
                
                # Use nearest heading as sheet name or fallback
                title = f"Table_{idx+1}"
                prev = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                sheet_name = "hello"
                title=["table","Champions -prix", "Driver Points", "Driver Stats", "Constructor points","Race"]
                df.to_excel(writer, sheet_name=title[i], index=False)
                i+=1
        output.seek(0)
        return send_file(output, as_attachment=True, download_name='F1.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return '''
    <form method="post">
        <label>Enter URL:</label>
        <input type="text" name="url" required style="width: 400px;">
        <input type="submit" value="Download Tables as Excel">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
