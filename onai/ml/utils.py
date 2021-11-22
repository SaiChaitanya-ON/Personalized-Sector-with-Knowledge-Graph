import functools
import tarfile
import tempfile

import fsspec
from transformers import DistilBertConfig, DistilBertTokenizer

from onai.ml.tools.torch.functional import pad_and_stack_tensor


def deep_get(dictionary, *keys):
    return functools.reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def freeze_distilbert(output_p, pretrained_distilbert):
    distill_bert_cfg_path = output_p / "distill_bert_cfg.json"
    with fsspec.open(distill_bert_cfg_path, "w") as fout:
        fout.write(
            DistilBertConfig.from_pretrained(pretrained_distilbert).to_json_string()
        )
    distill_bert_tokeniser_p = output_p / "tokeniser.tar.gz"
    with tempfile.TemporaryDirectory() as temp_dir, fsspec.open(
        distill_bert_tokeniser_p, "wb"
    ) as fout, tarfile.open(fileobj=fout, mode="w:gz") as tar_out:
        DistilBertTokenizer.from_pretrained(pretrained_distilbert).save_pretrained(
            temp_dir
        )
        tar_out.add(temp_dir, "")


def distilbert_process_text(texts, tokeniser, max_seq_length, pad_token_id, cuda):
    outs = tokeniser.batch_encode_plus(
        texts,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        padding=True,
    )
    input_ids = pad_and_stack_tensor(
        list(outs["input_ids"]), pad_value_by_last=False, pad_value=pad_token_id
    )
    attention_mask = pad_and_stack_tensor(
        list(outs["attention_mask"]), pad_value_by_last=False, pad_value=0
    )
    if cuda:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    return attention_mask, input_ids
