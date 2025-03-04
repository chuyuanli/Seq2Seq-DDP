from transformers import AutoModel, AutoTokenizer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache HuggingFace models locally")
    parser.add_argument("--model_name", type=str, default="t5-base",
                       help="Name of the model to download from HuggingFace (e.g. t5-base)")
    
    model_name = parser.parse_args().model_name

    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("Download complete! ------------------------------------------------------------")
    print("usually you should see the model and tokenizer in the .cache/huggingface/hub directory")

    