"""Build drafter-only vocabulary from GSM8K TRAIN; never inspect test answers."""
import argparse,json,hashlib,collections,os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
rows=[json.loads(l) for l in a.data.read_text().splitlines() if l.strip()];count=collections.Counter()
for i in range(0,len(rows),128):
    texts=[r['question']+'\n'+r['answer'] for r in rows[i:i+128]]
    for ids in tok(texts,add_special_tokens=False)['input_ids']:count.update(ids)
formatting='\n\n### Step 1: Let us solve this step by step. Therefore, the total is. Final Answer:\nAnswer: \\boxed{42} $$ \\frac{1}{2} \\text{ units} \\times \\div \\left \\right ** = + - / ( ) [ ] : 0 1 2 3 4 5 6 7 8 9 10 100 %'
mandatory=list(dict.fromkeys(tok.encode(formatting,add_special_tokens=False)+tok.all_special_ids))
ordered=mandatory+[i for i,_ in count.most_common() if i not in set(mandatory)]
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(dict(source=str(a.data.resolve()),source_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),train_rows=len(rows),
    tokenizer=a.tokenizer,mandatory_format_ids=mandatory,ids=ordered,counts={str(k):v for k,v in count.items()},
    note='Drafter-only shortlist; target always uses full vocabulary; test rows were not read'),indent=2))
print(json.dumps(dict(train_rows=len(rows),unique=len(ordered),coverage={k:sum(count[i] for i in ordered[:k])/sum(count.values()) for k in [2048,4096,8192,16384]})))
