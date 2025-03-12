# %%
import torch.cuda


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import LlamaForCausalLM

    # 选择合适的设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # 加载模型和 tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(
        device
    )

    def calculate_perplexity(model, tokenizer, text):
        # Tokenize 输入文本
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, sjq_sparse=2)

        # 计算 loss（交叉熵）
        loss = outputs.loss
        print(loss)
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
