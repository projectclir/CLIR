import gradio as gr
import pandas as pd
import some
from flask import Flask, render_template, url_for, redirect
#import ui





def retrieve_documents(Query, rcount):
        results = some.transs(Query, rcount)
        # Extract document ids and relevance scores from the results
        documents = []
        buttons = []
        for result in results:
            dummy = "View Document with ID"
            document = {"Document ID": result[0], "Relevance Score": round(result[1], 3)}
            documents.append(document)
            base_url = 'http://127.0.0.1:8080/{}'
            url = base_url.format(result[0])
            button = f'<a href={url} target="_blank">{dummy} {result[0]}</a>'
            buttons.append(button)
        df = pd.DataFrame(documents)
        buttons_html = "<br>".join(buttons)
        return df, buttons_html

app = Flask(__name__)

# @app.route('/')
# def home():
#     return ui.iface.launch(share=True)

@app.route('/<document_id>')
def serve_document(document_id):
    # Fetch document content using some.fetch_document_content(document_id)
    document_content = some.fetch_document_content(document_id)
    # Return document content as an HTML response
    return render_template('document.html',content=document_content)

if __name__ == '__main__':
    iface = gr.Interface(
        fn=retrieve_documents,
        inputs=[gr.inputs.Textbox(label="Query"), gr.inputs.Slider(minimum=1, maximum=10, step=1, label="Result Count")],
        outputs=[gr.outputs.Dataframe(type="pandas"), gr.outputs.HTML(label="Content")],
        allow_flagging="never",
        description='Enter your query and result count below:'
    )
    iface.launch(server_port=8080)
    
    app.run(port=8081)
