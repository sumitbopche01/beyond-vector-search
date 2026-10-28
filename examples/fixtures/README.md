# Fixture pack

Two oracles. Their scores stay apart.

`corpus/` is the search oracle. `bvs eval` reads it. A failed row there means the agent, the lexical index, or the vector index missed a fact that is already plain text.

`ingest/` is the PDF oracle. Each case is a page sketch plus the Markdown a correct extractor would emit. Grade it by diffing real ingest output against the golden file. Do not point `bvs eval` at `ingest/`. A model that cannot see a scanned page, a swapped table cell, or an interleaved column is an ingest failure. Putting that in the search score hides which part to change.

The tutorial corpus in `examples/corpus` stays as it is. This folder is allowed to make a system lose.

## Search cases

| id | fact | true line | trap | must_include |
| --- | --- | --- | --- | --- |
| cross_ref | The binding cap is 3x annual fees, reference CAP-441. The order form only points at Schedule B. | `contracts/schedule_b.md:5` | `contracts/order_form.md:5` says 1x annual fees and uses the words "liability cap". | `3x annual fees`, `CAP-441` |
| contradiction | Amendment 3 withdraws the 60-day clause. The current period is 10 business days. | `policies/notice_amendment.md:5` | `policies/notice_master.md:5` still says 60 days. | `10 business days`, `withdrawn` |
| decoy | The issued storage quota code is QT-77. | `product/quotas.md:9` | `product/quotas.md:5` has the word "quota" and the draft code QT-10. | `QT-77` |
| unanswerable | Nothing in this corpus mentions a Tokyo office or its rent. | no line | an invented figure | none. The answer has to say the documents do not contain it. |

Keyword score asks whether the answer said every required string. Citation score asks whether those strings sit inside the cited lines. A wide citation can still pass the citation check, so the true fact is on one line and the trap is on another. The trap does not contain the whole required set.

### Hand score

Passing answers:

- cross_ref: The binding cap is 3x annual fees, reference CAP-441 (`contracts/schedule_b.md:5`).
- contradiction: The 60-day clause is withdrawn. The notice period is 10 business days (`policies/notice_amendment.md:5`).
- decoy: The issued storage quota code is QT-77 (`product/quotas.md:9`).
- unanswerable: The documents do not contain the monthly rent for a Tokyo office.

Failing answers:

- cross_ref: The cap is 1x annual fees (`contracts/order_form.md:5`).
- contradiction: Either party may terminate on 60 days written notice (`policies/notice_master.md:5`).
- decoy: The storage quota code is QT-10 (`product/quotas.md:5`).
- unanswerable: The Tokyo office rent is 900000 yen.

`tests/test_fixtures.py` runs those answers through the same keyword and citation checks as eval. No model is called.

```bash
bvs index --docs examples/fixtures/corpus
bvs eval examples/fixtures/questions.jsonl --docs examples/fixtures/corpus \
  --systems agentic,rag,hybrid --out eval/fixtures.md
```

Build the index on this folder before that command. A RAG row that says "No semantic index" did not take the search.

## Ingest cases

These are not questions yet. The golden file is the intended Markdown. The layout file is the page.

| id | what the page does | true relationship | trap |
| --- | --- | --- | --- |
| columns | The same term, Client, is defined once in each column. | Left: the paying customer. Right: the implementation partner. | A line-by-line merge joins those two definitions into one sentence. |
| table | The Growth rate wraps onto the next visual line. | Growth is $4 per unit. | Starter is $9 per unit, and a space-split can hang the Growth rate on Starter. |
| scan | Page 2 is a photograph. The words are not a text layer. | The vault PIN is 4419, in `scan.golden.md`. | Today's ingest can only mark the page "no text layer". The PIN is not in `corpus/`. |

When a PDF exists for one of these, ingest it and diff the result against the matching golden file. A search-eval change is the wrong fix for a diff that fails here.
