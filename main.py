import sys
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
import torch.nn.functional as F
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from llama_cpp import Llama
from tkinter import filedialog, Tk
import gradio as gr
from functools import lru_cache
import html
import threading
import queue
import time

# Global variables to store model and tokenizer
model = None
cache = None
tokenizer = None
backend = "exllamav2"  # Default backend
max_context = 1024

tk_root = None
tk_queue = queue.Queue()

def tk_thread_func():
    global tk_root, tk_queue

    # Use tkinter to open a file dialog for model selection
    tk_root = Tk()
    tk_root.withdraw()  # Hide the root window
    tk_root.attributes('-topmost', True)  # Bring the file dialog to the front

    def process_tk_queue():
        try:
            task = tk_queue.get_nowait()  # Non-blocking
            task()
        except queue.Empty:
            pass
        tk_root.after(100, process_tk_queue)

    process_tk_queue()
    tk_root.mainloop()

tk_thread = threading.Thread(target=tk_thread_func)
tk_thread.daemon = True
tk_thread.start()

def unload_model():
    global model, cache, tokenizer
    if hasattr(model, "unload"):
        model.unload()
    del model
    model = None
    del cache
    cache = None
    del tokenizer
    tokenizer = None
    __import__("gc").collect()
    torch.cuda.empty_cache()
    return "Not loaded."

def load_model(model_path):
    global model, cache, tokenizer, backend
    unload_model()

    max_batch_size = 1

    if backend == "exllamav2":
        # Load model
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_batch_size = max_batch_size
        config.max_seq_len = max_context
        config.max_input_len = min(config.max_seq_len, 2048)
        config.max_attn_size = min(config.max_seq_len, 2048)**2
        #config.no_flash_attn = True

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True, batch_size=max_batch_size)
        model.load_autosplit(cache)
        tokenizer = ExLlamaV2Tokenizer(config)
    elif backend == "llama-cpp-python":
        model = Llama(model_path=model_path, n_ctx=max_context, use_mmap=False, logits_all=True, verbose=False)

def tokenize(text: str) -> list[int]:
    if backend == "exllamav2":
        return tokenizer.encode(text, add_bos=True)
    # For llama-cpp-python, tokenization would be handled internally, but for consistency:
    elif backend == "llama-cpp-python":
        return [model.tokenize(("\u25CF"+text).encode('utf-8'), add_bos=True)]

@lru_cache(maxsize=None)
def detokenize(ids: tuple[int, ...], heal_cp: bool = False) -> str:
    if backend == "exllamav2":
        return tokenizer.decode(torch.tensor(ids))
    elif backend == "llama-cpp-python":
        return model.detokenize(list(ids)).decode('utf-8', errors='strict' if heal_cp else 'replace')

def decode_tokens(ids: list[int], heal_cp: bool = False):
    if backend == "exllamav2":
        id_to_piece = tokenizer.get_id_to_piece_list()
        #if not heal_cp or not any('�' in id_to_piece[id] for id in ids):
        return [id_to_piece[id] for id in ids]
    elif backend == "llama-cpp-python":
        try:
            return [detokenize(tuple([id]), heal_cp) for id in ids]
        except UnicodeDecodeError:
            buffer = []
            output = []
            for id in ids:
                buffer.append(id)
                try:
                    output.append(detokenize(tuple(buffer), True))
                    buffer.clear()
                except UnicodeDecodeError:
                    output.append("\u200b")
                    continue
            return output

def get_token_perplexities(text: str) -> list[tuple]:
    if backend == "exllamav2":
        tokens = tokenize(text)
        tokens = tokens.squeeze(0)[:1024].unsqueeze(0)
        with torch.no_grad():
            all_logits = model.forward(tokens)
    elif backend == "llama-cpp-python":
        tokens = tokenize(text)[0]
        model.reset()
        model.eval(tokens)
        tokens = torch.tensor([tokens])
        all_logits = torch.tensor(model.scores[:model.n_tokens, :])
        all_logits = all_logits.view(1, all_logits.shape[0], all_logits.shape[1])

    # Compute softmax for all logits at once
    probabilities = F.softmax(all_logits[:, :-1, :], dim=-1)

    # Get the top 8 token possibilities for all positions at once
    top_probs, top_indices = torch.topk(probabilities, 8, dim=-1)

    # Squeeze to remove the batch dimension
    top_probs = top_probs.squeeze(0)
    top_indices = top_indices.squeeze(0)

    # Get the probabilities of the actual tokens
    token_probs = probabilities[0, torch.arange(probabilities.size(1)), tokens[0][1:]]

    token_info = []
    for i in range(len(tokens[0]) - 1):
        current_top_tokens = decode_tokens(top_indices[i]) #[detokenize(tuple([idx.item()])) for idx in top_indices[i]]
        current_top_probs = top_probs[i]
        token_info.append((token_probs[i].item(), list(zip(current_top_tokens, current_top_probs.tolist()))))

    return token_info

def visualize_tokens_with_color(text: str, token_info: list[tuple]):
    tokens = decode_tokens(tokenize(text)[0], True)[1:]
    colored_tokens = []
    for token, (token_prob, top_8) in zip(tokens, token_info):
        max_p = max(p for _, p in top_8)
        min_p = min(p for _, p in top_8)
        token_prob = (token_prob - min_p) / (max_p - min_p)

        if token_prob < 0.1:
            color = "red"
        elif token_prob < 0.25:
            color = "orange"
        elif token_prob > 0.7:
            color = "green"
        else:
            color = "yellow"

        def replaceUnprintable(text):
            return text.replace(' ', '␣').replace('\t', '⇥').replace('\n', '↵')
        
        tooltip = f"Top 8 possibilities:\n" + "\n".join([f"{replaceUnprintable(t)}: {p:.4f}" for t, p in top_8])

        colored_tokens.append(f'<span style="color:{color}" title="{html.escape(tooltip)}">{html.escape(token)}</span>')
    
    return ''.join(colored_tokens)

def select_model():
    model_path = None

    # The file picker is run on the main thread.
    def file_picker():
        nonlocal model_path
        if backend == "exllamav2":
            model_path = filedialog.askdirectory(title="Select Model Folder")
        elif backend == "llama-cpp-python":
            model_path = filedialog.askopenfilename(title="Select Model File")
    tk_queue.put(file_picker)

    while model_path is None:
        time.sleep(0.1)

    if not model_path:
        return None
    
    load_model(model_path)
    return f"Model loaded from: {model_path}\n- n_ctx: {max_context}"

def analyze_text(text):
    if model is None:
        return "Please load a model first."
    
    token_info = get_token_perplexities(text)
    colored_text = visualize_tokens_with_color(text, token_info)
    return colored_text

# Gradio Interface
with gr.Blocks(css="#output-text { overflow-y: auto !important; white-space: pre-wrap; }", analytics_enabled=False) as demo:
    gr.Markdown("# Token Perplexity Visualizer")

    with gr.Row():  # Create a row for splitting the input and output areas
        with gr.Column(scale=1):  # Left side (input section)
            with gr.Row():
                with gr.Column(scale=1):
                    backend_dropdown = gr.Dropdown(["exllamav2", "llama-cpp-python"], label="Select Backend", value="exllamav2")
                with gr.Column(scale=1):
                    ctx_number = gr.Number(label="Context Size", value=1024)
            
            model_output = gr.Textbox(label="Model Status", value="Not loaded.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    load_model_button = gr.Button("Select Model")
                with gr.Column(scale=1):
                    unload_model_button = gr.Button("Unload Model")
            
            input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...", lines=10)
            analyze_button = gr.Button("Analyze")
        
        with gr.Column(scale=1):  # Right side (output section)
            gr.Markdown("### Output")
            output_box = gr.HTML(elem_id="output-text")

    # Model selection logic
    load_model_button.click(fn=select_model, outputs=model_output)
    unload_model_button.click(fn=unload_model, outputs=model_output)
    
    # Text analysis logic
    analyze_button.click(fn=analyze_text, inputs=input_text, outputs=output_box)

    backend_dropdown.change(lambda choice: setattr(sys.modules[__name__], 'backend', choice), inputs=backend_dropdown)
    ctx_number.change(lambda choice: setattr(sys.modules[__name__], 'max_context', choice), inputs=ctx_number)

# Launch the Gradio app
demo.launch()
