# Error Analysis

## Overall findings

BM25-RAG outperformed dense-RAG on this evaluation set. BM25 achieved 70% overall answer accuracy and 62.5% accuracy on answerable questions, compared with 40% and 25% for dense-RAG. Both systems achieved 100% accuracy on the two unanswerable questions. BM25 also produced a higher retrieval hit rate at five (50% versus 50% tied), mean recall at five (50% versus 45.83%), mean reciprocal rank (0.40 versus 0.25), and complete ground-truth retrieval rate (50% versus 37.5%).

## Representative successes

### BM25 exact-identifier retrieval

For Q03, BM25 retrieved `A23P0039_chunk_0001` at rank 1. Llama correctly described the complete impact sequence—pickup truck, airport perimeter fence, and berm—and cited the directly supporting chunk. This illustrates BM25's strength when questions contain distinctive investigation identifiers and exact terminology.

### Dense semantic retrieval

During development, the descriptive Wemindji question retrieved `A23Q0145_chunk_0001` at rank 1 with dense retrieval. The query described a Beech King Air striking a snow windrow without relying only on the occurrence ID. This illustrates dense retrieval's ability to match semantic descriptions to relevant passages.

### Appropriate abstention

Both systems correctly returned `I don't know.` for Q09 and Q10. These questions asked for a pilot's favourite food and a pickup truck's licence-plate number, neither of which appeared in the corpus. The systems therefore achieved 100% unanswerable-question accuracy and avoided unsupported answers.

## Failure categories

### 1. Opaque identifiers harmed dense retrieval

Dense-RAG returned `I don't know.` for Q01 even though the answer existed in `A23Q0145_chunk_0001`. The query depended heavily on the arbitrary identifier `A23Q0145`, which has little semantic meaning. BM25 handled the identifier more effectively because it performs lexical term matching.

Potential improvement: combine lexical and dense scores in a hybrid retriever, or route identifier-heavy questions to BM25.

### 2. Multi-hop evidence coverage was insufficient

BM25 abstained on both multi-hop questions Q06 and Q07 because the top five results did not include all required chunks. Dense-RAG also abstained on Q07. Q06 required evidence from two separate runway-overrun reports and arithmetic over passenger counts. Dense retrieval returned some relevant evidence but not enough reliable context for a correct synthesis.

Potential improvement: retrieve per entity or investigation mentioned in the question, increase the candidate pool, diversify results across documents, and rerank candidates before generation.

### 3. Correct answers sometimes contained unsupported details

For BM25 Q02, the answer correctly identified the firefighting context and reforested landing area but introduced an unsupported 180-degree turn. The report did not contain this detail. This changed a partially correct response into an incorrect answer.

Potential improvement: use a stronger generation model, shorten the prompt, and add claim-by-claim verification against retrieved passages.

### 4. Citation formatting did not guarantee citation support

Several responses contained syntactically valid citations that did not support every factual claim. For Dense Q02, the cited chunk supported the forced landing but not the full operational context. For Dense Q04, the citation supported the stopping location and lack of injuries, but not added maintenance details. BM25 Q01 cited a chunk that had not been retrieved, which the automatic validator detected.

Potential improvement: validate that cited passages contain the named entities and details in the answer, and run a separate citation-selection or entailment step after generation.

### 5. Dense multi-hop generation produced factual and arithmetic errors

For Dense Q06, the model omitted the pilot when counting the seven occupants of A23C0081, calculated a difference of 50 instead of 49, and falsely reported injuries in A23O0046. It also cited an unrelated chunk that was not among the retrieved passages.

Potential improvement: require structured intermediate extraction for each report before comparison, then calculate totals programmatically rather than asking the LLM to perform all steps in one response.

## Evaluation limitations and possible bias

- The evaluation contains only 10 questions, so individual errors have a large effect on percentages.
- Several questions include investigation identifiers, favouring exact lexical matching and potentially disadvantaging dense retrieval.
- Questions are concentrated in a small subset of the 300-report corpus.
- Multi-hop questions explicitly name the reports being compared and may not represent all real user queries.
- The two unanswerable questions test clearly absent personal details; more subtle unanswerable questions could be harder.
- Chunk-level metrics require an exact match to the manually selected ground-truth chunks even when another chunk from the same report may contain overlapping evidence.
- Human correctness and citation judgments can be subjective, although a reference answer and explicit supporting chunks were used to improve consistency.

## Conclusion

The results show that retrieval quality strongly affects answer quality. BM25 was better suited to the identifier-heavy evaluation set, while dense retrieval worked well for semantically descriptive queries but struggled with arbitrary report codes. Llama generally abstained when evidence was insufficient, but citation presence alone did not guarantee that the cited passage supported the answer. Future work should focus on hybrid retrieval, multi-document evidence collection, reranking, and claim-level citation verification.
