import gradio as gr
import pandas as pd
import some

def retrieve_documents(Query, rcount):
        results = some.transs(Query, rcount)
        # Extract document ids and relevance scores from the results
        documents = []
        buttons = []
        for result in results:
            dummy = "View Document with ID"
            document = {"Document ID": result[0], "Relevance Score": round(result[1], 3)}
            documents.append(document)
            base_url = 'http://127.0.0.1:8066/{}'
            url = base_url.format(result[0])
            button = f'<a href={url} target="_blank">{dummy} {result[0]}</a>'
            buttons.append(button)
        df = pd.DataFrame(documents)
        buttons_html = "<br>".join(buttons)
        return df, buttons_html
iface = gr.Interface(
        fn=retrieve_documents,
        inputs=[gr.inputs.Textbox(label="Query"), gr.inputs.Slider(minimum=1, maximum=10, step=1, label="Result Count")],
        outputs=[gr.outputs.Dataframe(type="pandas"), gr.outputs.HTML(label="Content")],
        allow_flagging="never",
        description='Enter your query and result count below:')
iface.launch(share=True,server_port=8080)