# %%
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 选择合适的设备
    device = "mps"

    # 加载模型和 tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def calculate_perplexity(model, tokenizer, text):
        # Tokenize 输入文本
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        # 计算 loss（交叉熵）
        loss = outputs.loss
        ppl = torch.exp(loss)

        return ppl.item()

    # 测试文本
    text = "The quick brown fox jumps over the lazy dog."
    ppl = calculate_perplexity(model, tokenizer, text)
    print(f"Perplexity: {ppl}")


# %%
if __name__ == "__main__":
    main()
# %%
