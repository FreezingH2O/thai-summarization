import re
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


def normalize_th_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pythainlp/thaisum")
    p.add_argument("--model", default="google/mt5-small")
    p.add_argument("--max_input", type=int, default=256)
    p.add_argument("--max_target", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="./outputs")
    p.add_argument("--limit", type=int, default=500, help="smoke test size on Mac CPU")
    args = p.parse_args()

    print("Loading dataset:", args.dataset)
    ds = load_dataset(args.dataset)

    # Use train split (dataset might have only one split)
    train_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    def is_valid(ex):
        body = normalize_th_text(ex.get("body", ""))
        summ = normalize_th_text(ex.get("summary", ""))
        return len(body) > 80 and len(summ) > 10

    train_split = train_split.filter(is_valid)

    if args.limit and args.limit > 0:
        train_split = train_split.shuffle(seed=42).select(range(min(args.limit, len(train_split))))

    split = train_split.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(ex):
        title = normalize_th_text(ex.get("title", ""))
        body = normalize_th_text(ex.get("body", ""))
        summary = normalize_th_text(ex.get("summary", ""))

        inp = f"summarize th: {title} || {body}" if title else f"summarize th: {body}"

        model_inputs = tok(inp, max_length=args.max_input, truncation=True)
        labels = tok(summary, max_length=args.max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = [[(t if t != -100 else tok.pad_token_id) for t in l] for l in labels]
        pred_text = tok.batch_decode(preds, skip_special_tokens=True)
        label_text = tok.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(predictions=pred_text, references=label_text, use_stemmer=False)
        return {k: float(v) for k, v in scores.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=10,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("Saved model to:", args.out)


if __name__ == "__main__":
    main()
