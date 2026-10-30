# TRIPOD-LLM reporting checklist

> Aligned to the TRIPOD-LLM extension (Gallifant et al., Nat Med 2025). Item
> wording is paraphrased; attach the official checklist version at submission.
> Manuscript: `manuscript/JBI_submission_v1.md` (sections in brackets).

## 1. Title and abstract

| Item | Reported | Where |
|---|---|---|
| Identify the study as an evaluation of LLMs and state the clinical domain | Yes | Title; Abstract |
| Structured summary of objective, methods, results, conclusion | Yes | Abstract (Objective/Methods/Results/Conclusion) |

## 2. Introduction

| Item | Reported | Where |
|---|---|---|
| Scientific background and rationale | Yes | §1 |
| Study objectives and pre-specified questions | Yes | §1 (three questions) |
| Statement of significance / contribution | Yes | §1, Statement of Significance |

## 3. Methods

| Item | Reported | Where |
|---|---|---|
| Data sources and corpus; inclusion/exclusion of sources | Yes | §3.1; NCCN exclusion stated |
| Benchmark construction and item verification | Yes | §3.2 |
| Reference standard (gold answers) and evidence anchoring | Yes | §3.2 (verbatim quotes, adversarial audit) |
| Expert review: number, expertise, independence, blinding, adjudication | Yes | §3.2; Table 1 |
| Item-level inclusion/exclusion and difficulty screening | Yes | §3.2 (hard subset; v0.2 changes) |
| LLM characteristics: model names, versions, provider, access date | Partly | §3.4; exact API snapshots to be added by authors |
| Prompt templates, system messages, output format | Yes | §3.4; code in release |
| Decoding parameters (temperature, max tokens) and retry/truncation handling | Yes | §3.4 |
| Retrieval system, corpus, query construction, k, recall audit | Yes | §3.3 |
| Evaluation conditions and pairing | Yes | §3.3 |
| Outcomes and metrics (accuracy, S1 error, harm-weighted, trajectory metrics) | Yes | §3.5 |
| Statistical methods: CIs, paired tests, clustering, multiplicity, length control | Yes | §3.5 |
| Human baseline: sampling, blinding, instruction, analysis | Yes | §3.5, §4.6 |
| Automated scoring (rubric judge) and calibration against clinicians | Yes | §3.5, §4.5 |
| Sample size / power / MDE | Yes | §3.5; Supplementary MDE table |
| Missing data and parse failures | Yes | §3.4–3.5 |
| Fairness, equity and subgroup considerations | Partly | §5 (limitations: single-centre baseline, Chinese/English corpus); no demographic subgroup analysis |
| Ethics approval and consent | Template | Declarations (number to be inserted) |
| Data and code availability | Yes (template) | Declarations; release package |

## 4. Results

| Item | Reported | Where |
|---|---|---|
| Flow of items through construction, review and exclusion | Yes | Fig. 1; Table 1 |
| Model performance overall and by condition | Yes | Table 2; Fig. 2 |
| Performance by task and safety severity | Yes | Table 2; Supplementary task/MDE tables |
| Paired effects with uncertainty and multiplicity control | Yes | Table 2; Fig. 3 |
| Retrieval quality (recall@k) alongside RAG performance | Yes | §4.3; Supplementary Fig. S1 |
| Human baseline and human–model comparison | Yes | Table 4; §4.6 |
| Automated scoring reliability and human–machine agreement | Yes | §4.5; Supplementary Fig. S2 |
| Error analysis / failure taxonomy | Yes | §4.3–4.5; manuscript Fig. 2 (failure heatmap in supplement) |
| External / held-out model results | Yes | §4.2, §4.7 |
| Sensitivity analyses (v0.2, length control, expert-endorsed subset) | Yes | §4.7, §3.5, §4.1 |

## 5. Discussion

| Item | Reported | Where |
|---|---|---|
| Principal findings | Yes | §5 |
| Comparison with prior work | Yes | §5 |
| Implications for practice and research | Yes | §5 |
| Limitations, including saturation, selection, model panel, data sources | Yes | §5 |
| Generalizability | Yes | §5 |

## 6. Other information

| Item | Reported | Where |
|---|---|---|
| Funding | Template | Declarations |
| Conflicts of interest | Yes | Declarations |
| Author contributions (CRediT) | Template | Declarations |
| Acknowledgements | Yes | Declarations |
| Registration / protocol | Not applicable | No prospective registration; analysis plan pre-specified in code |
| Generative AI use in writing | Yes | Declarations |

## Items to complete before submission

1. Exact model snapshot identifiers and access dates.
2. Ethics approval/exemption number.
3. Repository URL and Zenodo DOI.
4. Funding and CRediT statements.
5. Attach the official TRIPOD-LLM checklist PDF with item numbers.
