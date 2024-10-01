import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from typing import Any
import torch
import torch.nn.functional as F
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from tkinter import filedialog, Tk
import gradio as gr
from functools import lru_cache
import html
import threading
import queue
import time
import gc
from dataclasses import dataclass

@dataclass
class LLMModel:
    model: Any = None
    cache: Any = None
    tokenizer: Any = None
    model_path: str = None
    backend: str = "transformers"
    max_context: int = 1024
    quant: str | None = None
    bos_token_id: int = -1

    def __hash__(self):
        return hash((self.model_path, self.backend))

    def unload_model(self):
        if hasattr(self.model, "unload"):
            self.model.unload()
        del self.model
        self.model = None
        del self.cache
        self.cache = None
        del self.tokenizer
        self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        self.unload_model()

        max_batch_size = 1

        if self.backend == "exllamav2":
            # Load model
            config = ExLlamaV2Config()
            config.model_dir = self.model_path
            config.prepare()
            config.max_batch_size = max_batch_size
            config.max_seq_len = self.max_context
            config.max_input_len = min(config.max_seq_len, 2048)
            config.max_attn_size = min(config.max_seq_len, 2048)**2
            #config.no_flash_attn = True

            self.model = ExLlamaV2(config)
            self.cache = ExLlamaV2Cache(self.model, lazy=True, batch_size=max_batch_size)
            self.model.load_autosplit(self.cache)
            self.tokenizer = ExLlamaV2Tokenizer(config)
            self.bos_token_id = self.tokenizer.bos_token_id
        elif self.backend == "llama-cpp-python":
            self.model = Llama(model_path=self.model_path, n_ctx=self.max_context, use_mmap=False, logits_all=True, verbose=False)
            self.bos_token_id = self.model.token_bos()
        elif self.backend == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.bos_token_id = self.tokenizer.bos_token_id

            kwargs = {}
            if self.quant == "8bit":
                kwargs = {"load_in_8bit": True}
            elif self.quant == "4bit":
                kwargs = {"load_in_4bit": True}

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", **kwargs)
            self.model.eval()

        if self.bos_token_id is None:
            token = self.tokenize("\u25CF", add_bos=False)[0]
            assert len(token) == 1, token
            self.bos_token_id = token[0]

    def tokenize(self, text: str, add_bos=True) -> list[int]:
        # For now, it's required that some BOS token is always present in the output of this function when add_bos is True.
        # This may not be optimal as some models don't need the BOS to be present.
        if self.backend == "transformers":
            tokenized_text = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            if add_bos:
                bos_token_tensor = torch.tensor([self.bos_token_id])
                return torch.cat((bos_token_tensor, tokenized_text.squeeze(0)), dim=0).unsqueeze(0)
            else:
                return tokenized_text
        elif self.backend == "exllamav2":
            tokenized_text = self.tokenizer.encode(text, add_bos=False)
            if add_bos:
                bos_token_tensor = torch.tensor([self.bos_token_id])
                return torch.cat((bos_token_tensor, tokenized_text.squeeze(0)), dim=0).unsqueeze(0)
            else:
                return tokenized_text
        elif self.backend == "llama-cpp-python":
            return [([self.bos_token_id] if add_bos else []) + self.model.tokenize(text.encode('utf-8'), add_bos=False)]

    @lru_cache(maxsize=None)
    def detokenize(self, ids: tuple[int, ...], heal_cp: bool = False) -> str:
        if self.backend == "transformers":
            return self.tokenizer.decode(torch.tensor(ids))
        elif self.backend == "exllamav2":
            return self.tokenizer.decode(torch.tensor(ids))
        elif self.backend == "llama-cpp-python":
            return self.model.detokenize(list(ids)).decode('utf-8', errors='strict' if heal_cp else 'replace')

    def decode_tokens(self, ids: list[int], heal_cp: bool = False):
        if self.backend == "transformers":
            return [self.tokenizer.decode([id]) for id in ids]
        elif self.backend == "exllamav2":
            id_to_piece = self.tokenizer.get_id_to_piece_list()
            return [id_to_piece[id] for id in ids]
        elif self.backend == "llama-cpp-python":
            try:
                return [self.detokenize(tuple([id]), heal_cp) for id in ids]
            except UnicodeDecodeError:
                buffer = []
                output = []
                for id in ids:
                    buffer.append(id)
                    try:
                        output.append(self.detokenize(tuple(buffer), True))
                        buffer.clear()
                    except UnicodeDecodeError:
                        output.append("\u200b")
                        continue
                return output
            
    def get_token_perplexities(self, text: str, topk: int) -> list[tuple]:
        if self.backend == "transformers":
            tokens = self.tokenize(text)
            tokens = tokens.squeeze(0)[:self.max_context].unsqueeze(0)
            with torch.no_grad():
                all_logits = self.model.forward(tokens).logits
        elif self.backend == "exllamav2":
            tokens = self.tokenize(text)
            tokens = tokens.squeeze(0)[:self.max_context].unsqueeze(0)
            with torch.no_grad():
                all_logits = self.model.forward(tokens, cpu_logits=True)
        elif self.backend == "llama-cpp-python":
            tokens = self.tokenize(text)[0][:self.max_context]
            self.model.reset()
            self.model.eval(tokens)
            tokens = torch.tensor([tokens])
            all_logits = torch.tensor(self.model.scores[:self.model.n_tokens, :])
            all_logits = all_logits.view(1, all_logits.shape[0], all_logits.shape[1])

        probabilities = F.softmax(all_logits[:, :-1, :], dim=-1)
        top_probs, top_indices = torch.topk(probabilities, topk, dim=-1)
        top_probs = top_probs.squeeze(0)
        top_indices = top_indices.squeeze(0)
        token_probs = probabilities[0, torch.arange(probabilities.size(1)), tokens[0][1:]]

        token_info = []
        for i in range(len(tokens[0]) - 1):
            current_top_tokens = self.decode_tokens(top_indices[i])
            current_top_probs = top_probs[i]
            token_info.append((token_probs[i].item(), list(zip(current_top_tokens, current_top_probs.tolist()))))

        return token_info

# Global variables to store model and tokenizer
models: list[LLMModel] = [LLMModel(), LLMModel()]
topk: int = 10

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

def get_tokens_with_color(model: LLMModel, text: str):
    token_info = model.get_token_perplexities(text, topk)
    tokens = model.decode_tokens(model.tokenize(text)[0][:model.max_context], True)[1:]
    
    colored_tokens = []
    for token, (token_prob, top) in zip(tokens, token_info):
        max_p = max(p for _, p in top)
        min_p = min(p for _, p in top)
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
        
        tooltip = f"Top {topk} probabilities:\n" + "\n".join([f"{replaceUnprintable(t)}: {p:.4f}" for t, p in top])

        colored_tokens.append((token, color, tooltip))
    
    return colored_tokens

def select_model(idx: int, backend: str, max_context: int, quant: str, lazy_load: bool):
    model_path = None
    
    def file_picker():
        nonlocal model_path
        if backend == "exllamav2" or backend == "transformers":
            model_path = filedialog.askdirectory(title="Select Model Folder")
        elif backend == "llama-cpp-python":
            model_path = filedialog.askopenfilename(title="Select Model File")
    tk_queue.put(file_picker)

    while model_path is None:
        time.sleep(0.1)

    if not model_path:
        return None
    
    if backend != "transformers":
        quant = None
    
    model = models[idx]
    model.model_path = model_path
    model.backend = backend
    model.max_context = max_context
    model.quant = quant
    if not lazy_load:
        model.load_model()
    
    return f"Model {'loaded' if not lazy_load else 'selected'} from: {model_path}\n- Context Lenght: {max_context}" + (f"\n- Quant: {quant}" if quant else "")

def unload_model(idx):
    model = models[idx]
    model.unload_model()
    return "Not loaded."

def analyze_text(idx: int, text: str):
    global models

    model = models[idx]
    if model is None:
        return "Please load a model first."
    
    result = get_tokens_with_color(model, text)
    return ''.join([f'<span class="token" style="color:{color}" title="{html.escape(tooltip)}">{html.escape(token)}</span>' for token, color, tooltip in result])

def compare_models(text: str):
    if not models[0].model_path or not models[1].model_path:
        return "Please select both models first."
    
    models[0].load_model()
    result1 = get_tokens_with_color(models[0], text)
    models[0].unload_model()
    
    models[1].load_model()
    result2 = get_tokens_with_color(models[1], text)
    models[1].unload_model()

    # Generate HTML for Model 1 and Model 2
    html1 = ''.join([f'<span class="token" style="color:{color}" title="{html.escape(tooltip)}">{html.escape(token)}</span>' for token, color, tooltip in result1])
    html2 = ''.join([f'<span class="token" style="color:{color}" title="{html.escape(tooltip)}">{html.escape(token)}</span>' for token, color, tooltip in result2])

    if len(result1) != len(result2) or [t for t, _, _ in result1][-10:] != [t for t, _, _ in result2][-10:]:
        diff_html = ["Tokenizer mismatch, diff not available."]
    else:
        # Generate HTML for Diff
        diff_html = []
        for (token1, color1, _), (_, color2, _) in zip(result1, result2):
            if (color1 in ["red", "orange", "yellow"] and color2 in ["green"]):
                diff_color = "green"
            elif color1 in ["red"] and color2 in ["orange", "yellow"]:
                diff_color = "orange"
            elif color1 in ["green"] and color2 in ["red", "orange", "yellow"]:
                diff_color = "red"
            else:
                diff_color = "white"
            
            diff_html.append(f'<span class="token" style="color:{diff_color}">{html.escape(token1)}</span>')

    return html1, html2, ''.join(diff_html)

# Gradio Interface
with gr.Blocks(css=".prose.output-text { overflow-y: auto !important; white-space: pre-wrap; max-height: 80vh; } .token:hover { background-color: gray; }", analytics_enabled=False) as demo:
    gr.Markdown("# Token Perplexity Visualizer")

    with gr.Tabs():
        with gr.TabItem("Single Model Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            backend_dropdown = gr.Dropdown(["transformers", "exllamav2", "llama-cpp-python"], label="Backend", value=models[0].backend)
                            ctx_number = gr.Number(label="Context Size", value=models[0].max_context)
                            backend_quant_radio = gr.Radio(["None", "8bit", "4bit"], label="Quantization", value="None", visible=(models[0].backend == "transformers"))
                            
                            def update_visibility(dropdown):
                                value = dropdown
                                return gr.Radio(visible=(value == "transformers"))

                            backend_dropdown.change(update_visibility, backend_dropdown, backend_quant_radio)
                    
                    model_output = gr.Textbox(label="Model Status", value="Not loaded.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            load_model_button = gr.Button("Load Model")
                        with gr.Column(scale=1):
                            unload_model_button = gr.Button("Unload Model")
                    
                    input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...", lines=10)
                    analyze_button = gr.Button("Analyze")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    output_box = gr.HTML(elem_classes="output-text")

            # Model selection logic
            load_model_button.click(fn=lambda _1, _2, _3: select_model(0, _1, _2, _3, False), inputs=[backend_dropdown, ctx_number, backend_quant_radio], outputs=model_output)
            unload_model_button.click(fn=lambda: unload_model(0), outputs=model_output)
            
            # Text analysis logic
            analyze_button.click(fn=lambda _1: analyze_text(0, _1), inputs=input_text, outputs=output_box)

        with gr.TabItem("Model Comparison"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    backend_dropdown1 = gr.Dropdown(["transformers", "exllamav2", "llama-cpp-python"], label="Model 1 Backend", value=models[0].backend)
                                    ctx_number1 = gr.Number(label="Model 1 Context Size", value=models[0].max_context)
                                    backend_quant_radio1 = gr.Radio(["None", "8bit", "4bit"], label="Quantization", value="None", visible=(models[0].backend == "transformers"))
                        
                                def update_visibility(dropdown):
                                    value = dropdown
                                    return gr.Radio(visible=(value == "transformers"))

                                backend_dropdown1.change(update_visibility, backend_dropdown1, backend_quant_radio1)
                            
                            model_output1 = gr.Textbox(label="Model 1 Status", value="Not loaded.")
                            
                            load_model_button1 = gr.Button("Select Model 1")
                            
                        with gr.Column(scale=1):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    backend_dropdown2 = gr.Dropdown(["transformers", "exllamav2", "llama-cpp-python"], label="Model 2 Backend", value=models[1].backend)
                                    ctx_number2 = gr.Number(label="Model 2 Context Size", value=models[1].max_context)
                                    backend_quant_radio2 = gr.Radio(["None", "8bit", "4bit"], label="Quantization", value="None", visible=(models[1].backend == "transformers"))
                        
                                def update_visibility(dropdown):
                                    value = dropdown
                                    return gr.Radio(visible=(value == "transformers"))

                                backend_dropdown2.change(update_visibility, backend_dropdown2, backend_quant_radio2)
                            
                            model_output2 = gr.Textbox(label="Model 2 Status", value="Not loaded.")
                            
                            load_model_button2 = gr.Button("Select Model 2")

                    compare_input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...", lines=10)
                    compare_button = gr.Button("Analyze Both")

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("Model 1 Output"):
                            model1_output = gr.HTML(elem_classes="output-text")
                        with gr.TabItem("Model 2 Output"):
                            model2_output = gr.HTML(elem_classes="output-text")
                        with gr.TabItem("Diff"):
                            output_box_diff = gr.HTML(elem_classes="output-text")

            # Model 1 selection logic
            load_model_button1.click(fn=lambda _1, _2, _3: select_model(0, _1, _2, _3, True), inputs=[backend_dropdown1, ctx_number1, backend_quant_radio1], outputs=model_output1)

            # Model 2 selection logic
            load_model_button2.click(fn=lambda _1, _2, _3: select_model(1, _1, _2, _3, True), inputs=[backend_dropdown2, ctx_number2, backend_quant_radio2], outputs=model_output2)

            compare_button.click(fn=compare_models, inputs=compare_input_text, outputs=[model1_output, model2_output, output_box_diff])

# Launch the Gradio app
demo.launch()
