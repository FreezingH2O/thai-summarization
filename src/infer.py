import argparse
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs_mac_smoke")
    p.add_argument("--title", default="")
    p.add_argument("--body", required=True)
    args = p.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained("google/mt5-small")


    title = norm(args.title)
    body = norm(args.body)

    inp = f"summarize th: {title} || {body}" if title else f"summarize th: {body}"

    x = tok(inp, return_tensors="pt", truncation=True, max_length=512)
    y = model.generate(
    **x,
    max_new_tokens=48,
    num_beams=6,
    no_repeat_ngram_size=3,
    length_penalty=0.8,
    early_stopping=True,
    bad_words_ids=[[tok.convert_tokens_to_ids("<extra_id_0>")]],

)

    out = tok.decode(y[0], skip_special_tokens=True)
    out = out.replace("<extra_id_0>", "").strip()
    print(out)



if __name__ == "__main__":
    main()
