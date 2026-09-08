# `tex-submission`: a sentence-by-sentence revision plan

**Date:** 2026-08-28  
**Status:** review and plan only; no paper source file has been modified yet  
**Target:** the concision, cross-disciplinary clarity and logical coherence of a flagship journal such as `Nature`

## 0. The strict scope of this plan

Review only the paper text actually included in the current submission package:

- `tex-submission/main.tex`
- `tex-submission/front_meta.tex`
- `tex-submission/front_body.tex`
- `tex-submission/body.tex`
- `tex-submission/methods.tex`
- `tex-submission/si.tex`
- `tex-submission/si_body.tex`

Do not review the cover letter, the old `front.tex`, the reference-verification record, other directories, the fact ledger, the code, the data or the DFT records.

This plan addresses three kinds of problem only:

1. wordiness or repetition;
2. passages a reader outside the field will find hard to follow;
3. incoherence between sentences, between paragraphs or between sections.

The authors ask that every strong conclusion and headline result be kept. This plan contains no suggestion to soften, narrow or de-overclaim any conclusion.

## 1. Notation for the actions

- `SPLIT`: break one long multi-proposition sentence into two or three.
- `MERGE`: combine adjacent repetitive sentences, keeping the information once.
- `CUT`: delete a repetitive sentence, a piece of metadiscourse or an empty connective that adds no function.
- `MOVE`: move a sentence to a more suitable paragraph, or next to the evidence it rests on.
- `REWRITE`: keep the facts and the conclusion, rewriting only the subject, predicate, reference or parallel structure.
- `BREAK`: keep the sentence order but establish a new controlling idea for a paragraph.
- `KEEP`: no change needed.

---

# A. The main text, sentence by sentence

## A1. `main.tex`

### Author information, keywords and statements

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| M-01 | `main.tex:50-51`, `crystal chemistry ... autonomous scientific discovery` | the six keywords span very different levels and their order does not reflect the paper's storyline | `REWRITE`: order them as "the core object PRIS/crystal chemistry -> the method, autonomous discovery -> the application, generative materials"; add and remove no scientific concept | brings the search-term order into line with the title and abstract |
| M-02 | `main.tex:71-75`, `Entries from ... are not redistributable ... and ELEMENTA ... The public benchmark ... therefore...` | one sentence carries two licence states and the source of the public benchmark | `SPLIT`: the first sentence states the ICSD and the ELEMENTA licences separately; the second states on its own that the public benchmark consists of COD alone | clearly separates "what may not be redistributed" from "what is actually released" |
| M-03 | `main.tex:75-76`, `Derived scalar features, split assignments and all numerical results...` | no evident problem | `KEEP` | states the scope of the Source Data |
| M-04 | `main.tex:76-80`, `The 260 design candidates ... are provided ... one CIF per candidate, with an index giving...` | five kinds of metadata are listed at the end and the main clause runs too long | `SPLIT`: the first sentence states only that 260 relaxed CIFs are provided; the second lists the composition, space group, site fraction, PSS and bulk modulus in the index | separates data availability from the index fields |
| M-05 | `main.tex:63-68` Acknowledgements and Competing interests | concise and clear | `KEEP` | unchanged |
| M-06 | `main.tex:82-84` Code availability | concise and clear | `KEEP` | unchanged |

## A2. `front_meta.tex`: title and abstract

### Title, lines 7-8

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| FM-01 | `front_meta.tex:7-8`, `Autonomous discovery of crystal-structure plausibility laws for explainable and rapid crystal screening and diagnosis` | `crystal` repeats, and there are too many stacked nouns and double modifiers | `REWRITE`: keep the three layers autonomous discovery, plausibility laws and screening/diagnosis, compress the repeated `crystal`, and make `rapid` and `explainable` modify parallel objects; check against the Nature title length at the end | one reading gives both "what was discovered" and "what it is for" |

### Abstract, lines 11-28

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| FM-02 | `11-12`, `faster than density functional theory (DFT) energy and phonon calculations, or experiments, can assess them` | the three modes of assessment are not syntactically parallel and the inserted commas impede reading | `REWRITE`: make DFT energy calculations, phonon calculations and experiments strictly parallel | establishes the tension between the speed of generation and the speed of assessment |
| FM-03 | `12-14`, `Deciding which merit expensive computation is therefore the bottleneck, yet most screens...` | one sentence gives both the bottleneck and the gap in existing methods | `SPLIT`: the first sentence states that prioritising candidates is the bottleneck; the second takes current rapid screens as its subject and states that they only exclude atomic overlap | moves naturally from the problem to the methodological gap |
| FM-04 | `14-16`, `Herein, agents propose, implement, test and actively refute...` | `Herein` is stiff and four action verbs are crowded together | `REWRITE`: open with `Here` or an explicit subject; compress the actions to "generate and actively test/refute", keeping the two million and the eight laws | states the scale of the method and the core finding concisely |
| FM-05 | `16-18`, `These laws encode five mechanisms: ...` | the list is clear, but its connection to the previous sentence could be tighter | `MERGE`: keep the full list of five mechanisms and make `These laws` refer explicitly to the eight PRIS laws; do not break up the list | explains the physico-chemical content of the eight laws |
| FM-06 | `18-19`, `Our laws keep 82--99% ... against 6.5%...` | `against` elides the second predicate of the comparison | `REWRITE`: write two syntactically symmetric clauses, each using keep/satisfy explicitly | the comparison is understood without rereading |
| FM-07 | `19-20`, `They also detect 87.9% of damage...` | `They` may refer to the eight individual laws or to all the law sets; `damage` is not made concrete at first mention | `REWRITE`: name the strictest PRIS set directly, and say controlled structural damage explicitly | identifies precisely what the 87.9% refers to |
| FM-08 | `20-21`, `Moreover, the plausibility they measure is proven to be linearly correlated...` | `Moreover` is empty and the passive slows the main conclusion | `REWRITE`: drop the empty connective and the passive, and state the strong linear-correlation conclusion directly with PRIS plausibility as the subject | moves from damage detection to synthesizability |
| FM-09 | `21-23`, `therefore explainably screens...` | `explainably` reads unnaturally and the reader does not learn what "explainable" means concretely | `REWRITE`: use `mechanism-resolved screening` or an equivalent, keeping the 83.7% and 80.7% | states that the explanatory power comes from naming the failure mechanism |
| FM-10 | `23-25`, `cut the DFT validation queue by up to 67.3% while keeping 99.2%...` | two headline numbers are crowded into one long sentence and the second is easily swallowed | `SPLIT`: the first sentence foregrounds the queue reduction; the second the 99.2% retention of target-reaching candidates; the tunability is explained in Results/Methods | foregrounds the saving and the retention separately |
| FM-11 | `25-26`, `explains why GNoME ... and why falsified crystals...` | two external cases of different kinds share one `why` construction | `SPLIT`: give the GNoME chemical ordering and the fixed-coordinate wrong-element diagnosis a parallel clause or short sentence each | each case carries one diagnostic function |
| FM-12 | `26-28`, `moves screening from pass-or-fail to why ... showing autonomous agents can discover...` | `from ... to why` is not parallel, and one sentence closes both PRIS and agentic science | `SPLIT`: the first sentence expresses the move from a binary verdict to a chemical diagnosis in a parallel structure; the second opens with `More broadly` and gives the strong conclusion about autonomous active refutation | closes the tool's value first, then the broader significance |

After these actions the abstract keeps its original order: `bottleneck -> autonomous discovery -> five mechanisms -> quantitative performance -> PSS/inverse design -> external cases -> broader significance`.

## A3. `front_body.tex`: Introduction

### Paragraph 1, lines 4-21: the screening bottleneck and the precision gap

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| I-01 | `5-11`, `High-throughput databases ... now supply candidates faster than they can be assessed, whether by...` | the subject list, the citations and the three modes of assessment nest inside one long sentence | `SPLIT`: the first sentence reports the rate of candidate generation; the second lists DFT energies, phonons and experiments in parallel | explains why validation is the bottleneck |
| I-02 | `12-13`, `An initial screen must therefore decide which predictions consume...` | repeats `deciding which predictions warrant...` from lines 4-5 | `MERGE`: fold the gatekeeper meaning into the end of the previous group and delete the repeated verb construction | states "speed mismatch -> the need for screening" once |
| I-03 | `13-17`, `Yet many generative pipelines test little more than... and avoiding gross overlap establishes neither...` | the current practice and its limitation are packed into one sentence | `SPLIT`: the first sentence states the fixed minimum distance; the second takes `Passing this test` as its subject and lists coordination, electrostatics, bond valence and ordering | says what is done first, then what is missed |
| I-04 | `17-18`, `Commentary now argues that data alone...` | opening in the style of a literature commentary interrupts the straight line from screening limitation to the gap this paper fills | `MERGE`: delete the `Commentary now argues` metadiscourse and merge "AI must learn chemical rules" with the gap in the next sentence, keeping the citation | leads directly to the need for interpretable chemical rules |
| I-05 | `19-20`, `rules ... that name the assumption a structure violates` | `name the assumption` is abstract | `REWRITE`: make it identify the physical or chemical constraint violated | defines concretely what "explainable" means here |
| I-06 | `20-21`, `Such rules would sit between...` | clear, and carries the key positioning of the layer | `KEEP` | closes paragraph 1 |

### Paragraph 2, lines 23-37: the opposite failures of Pauling and of a distance threshold

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| I-07 | `23-26`, `they connect ionic size ... and need only a radius table...` | the chemical content and the low computational cost are two propositions | `SPLIT`: the first sentence lists the four chemical ideas; the second states separately that only a radius table and formal charges are needed | presents the richness and the economy separately |
| I-08 | `26-28`, `They belong to a tradition of simple laws...` | a general evaluation standing as its own sentence adds little information | `MERGE`: fold it into the second sentence of I-07 as a summary of "chemical content + low input cost", keeping the citation | states the value of the Pauling precedent compactly |
| I-09 | `29-32`, the literature `13%`, the present `6.5%` and the distance `1.6/3.2%` | the literature result and this work's two results are stacked in succession | `SPLIT`: give the literature audit first; then this work's 6.5%; then introduce the distance threshold on its own with `At the opposite extreme` | forms a symmetric "too strict -- too loose" structure |
| I-10 | `32-34`, `The classical rules are ... and the distance cutoff is...` | the symmetry is clear | `KEEP`, checking only the length and parallel grammar of the two clauses | summarises the opposite failure modes in one sentence |
| I-11 | `34-36`, `How many experimental structures ... since a bound loose enough...` | the principle and the example nest inside one causal sentence | `SPLIT`: the first sentence states that retention alone cannot measure a screen; the second explains why through a very loose bound | gives an intuitive basis for the two-metric criterion |
| I-12 | `36-37`, `A good law must therefore...` | the three requirements are parallel and their function is clear | `KEEP` | arrives at the evaluation criterion |

### Paragraph 3, lines 39-46: two practical questions

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| I-13 | `39`, `Two further questions decide what such rules would be worth` | too abstract a signpost | `REWRITE`: name the two tests, synthesizability and structural correctness, directly | previews the two questions of this paragraph |
| I-14 | `41-45`, `Databases record ... so synthesizability was estimated ... none of which ...; whether...` | the data difficulty, the three classes of method and the open question are crowded into one long sentence | `SPLIT`: write in turn the database difficulty, the three surrogate methods, the fact that none reads a chemical bound, and this work's open question | advances as "difficulty -> current practice -> gap" |
| I-15 | `45-46`, `whether the structure handed downstream is correct at all, because every assessment...` | the question and the reason for its importance share a sentence, and `it` refers a long way back | `SPLIT`: the first sentence raises correctness; the second uses `the supplied structure` to state the assumption every later assessment makes | raises the second question clearly |

### Paragraph 4, lines 48-66: the GNoME/A-Lab and wrong-element cases

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| I-16 | `48`, `Prominent cases show that it need not be` | the antecedent of `it` is unclear | `REWRITE`: write directly that structures handed downstream can be chemically misassigned | follows on from structural correctness |
| I-17 | `48-51`, GNoME `381,000...421,000... enriched...` | the database scale and the low-symmetry phenomenon are crowded into one sentence | `SPLIT`: the first sentence gives the scale; the second states the rare low-symmetry enrichment and the passing of the distance filter | separates the background scale from the phenomenon to be explained |
| I-18 | `51-53`, `similar elements, such as ... ordered artificially...` | the element examples are inserted into the middle of the main clause | `REWRITE`: say first that similar elements are artificially ordered onto sites that should be equivalent, then give rare-earth/Zr-Hf as examples at the end | the mechanism is understood first, the chemical instances after |
| I-19 | `53-55`, `The same variable recurred at A-Lab...` | `same variable` is vague | `REWRITE`: use chemical ordering directly, and group it with the follow-up information into one A-Lab case | names the variable the two cases share |
| I-20 | `55-57`, `A reanalysis disputed ... and an author correction...` | two subsequent events are packed into one sentence | `SPLIT`: write the reanalysis and the author correction separately, keeping them contiguous with I-19 | sets out the evolution of the dispute clearly |
| I-21 | `58-60`, `documented at least 70 falsified structures that reused...` | a circumlocution | `REWRITE`: write directly that genuine diffraction data were paired with altered element identities | states the second failure mechanism quickly |
| I-22 | `60-61`, `An archive ... repeats this...` | `this` is vague and the standalone sentence carries too little | `MERGE`: fold it into I-21 as the note that the pattern repeats across the Cu/Ni/Mn/Fe labels | forms one complete historical-case sentence |
| I-23 | `61-62`, `These cases share one missing check ... examined by neither...` | an inserted appositive obscures the subject and predicate | `REWRITE`: take distance filters and energy calculations as the subject and say directly that they do not check chemical ordering or elemental identity | extracts the gap the two cases share |
| I-24 | `62-65`, `two open cases of different kinds: ...` | `different kinds` is empty and the two items after the colon are not parallel | `SPLIT`: two sentences, one for artificial ordering and one for a wrong occupant at plausible coordinates | sets up parallel problems for the two Results applications |
| I-25 | `65-66`, `Closing either requires laws...` | clear | `KEEP` | closes the cases into this work's requirement |

### Paragraph 5, lines 68-92: the research route and a preview of the results

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| I-26 | `68-71`, `Given experimental structures ... they proposed, implemented, tested and refuted...` | the inputs and four actions are crowded into one sentence | `SPLIT`: the first sentence lists the inputs; the second compresses to generated/implemented/actively tested candidates | says what the system was given, then what it did |
| I-27 | `71-72`, `Failed claims stayed on record...` | the causal relation to the previous sentence could be more direct | `MERGE`: attach it to the method sentence with `Because failed claims remained...` | explains what refutation-driven progress means |
| I-28 | `72-74`, `Two million candidate laws were reduced ... to eight...` | the strong conclusion is itself clear | `KEEP`; make sure the paragraph reports it in full only once | states the search funnel |
| I-29 | `74-76`, `Each law encodes ... five mechanisms ... so every violation...` | the mechanism list and the diagnostic consequence are two propositions | `SPLIT`: one sentence for the list; one for every violation pointing at a mechanism | separates "what it is" from "what it is good for" |
| I-30 | `76-78`, `kept 82--99% ... and the strictest detected 87.9%...` | two performance figures share a sentence | `SPLIT`: one sentence each for retention and damage detection | foregrounds the two side by side |
| I-31 | `78-79`, `No synthesis label ... yet the plausibility they measure...` | `their/they` refer loosely | `REWRITE`: name the PRIS laws and PRIS plausibility directly, keeping the strong linear-correlation conclusion | moves from structural performance to synthesizability |
| I-32 | `79-81`, `PSS screened 83.7% ... keeping 80.7% ... and every removal...` | two proportions and the explanatory result are crowded together | `SPLIT`: one sentence for performance; one for mechanism traceability | foregrounds performance and explanatory power separately |
| I-33 | `82-83`, `A hull-energy threshold reached 72.0%...` | inserting a secondary baseline at the end of the Introduction interrupts the storyline | `CUT`: remove it from the Introduction and keep the full comparison in Results | keeps the route paragraph from becoming a list of results |
| I-34 | `83-87`, the inverse-design task, `67.3%`, `260`, `99.2%`, relaxed cells | one sentence carries the task, the saving, the validation and the data availability | `SPLIT/COMPRESS`: the Introduction keeps only the task and the two headline results; the 260 calculations and the data availability go to Results and the statements | previews the application compactly without restating the method |
| I-35 | `87-89`, `The same mechanisms speak to both open cases...` | the figurative `speak to`, and both cases in one sentence | `REWRITE/SPLIT`: use applies mechanism-resolved diagnosis; give GNoME and the wrong element a sentence or a parallel clause each | refers back to the two earlier questions explicitly |
| I-36 | `89-90`, `turns ... from a pass-or-fail check into a chemical diagnosis` | the core sentence is clear | `KEEP` | closes the value of PRIS |
| I-37 | `90-92`, `It also shows...` | `It` refers unclearly | `REWRITE`: take `This active-refutation workflow` as the subject | lands precisely on the strong conclusion about autonomous science |

**Paragraph action:** `BREAK` before line 78. Lines 68-78 cover only the discovery process, the eight laws, the mechanisms and the structural performance; lines 78-92 cover synthesizability, inverse design, the two external cases and the overall significance.

## A4. `body.tex`: Results and Discussion

### Results subheadings

| no. | line and original heading | problem | specific action | function after the change |
|---|---|---|---|---|
| R-H1 | `8`, `Autonomous agents discover eight laws through proposal, test and refutation` | too long; three process nouns occupy the body of the heading | `REWRITE`: keep autonomous discovery, eight laws and active refutation, compressed into one short conclusion-bearing heading | the new finding of this section is visible at a glance |
| R-H2 | `66`, `PRIS balances experimental-structure satisfaction with damage detection` | accurate but too long | `REWRITE`: compress the compound nouns, keeping the satisfaction-detection balance | foregrounds the performance trade-off |
| R-H3 | `165`, `Five complementary mechanisms turn screening into diagnosis` | the core information is clear; only slightly long | `REWRITE`: remove modifiers the body text can carry, keeping five mechanisms and diagnosis | emphasises the physico-chemical significance |
| R-H4 | `235`, `PRIS screens candidates before expensive calculations` | clear | `KEEP`, checking it only when the subheading lengths are harmonised at the end | introduces the application |
| R-H5 | `381`, `Plausibility precedes stability checks and addresses controversy and identity failure in crystallography` | three conclusions at once, and far longer than the other headings | `REWRITE`: the heading keeps one controlling idea, either ordering/identity diagnosis or plausibility-before-stability; the other becomes the topic sentence at line 386 | lets the section answer one overall question |

### Results 2.1, lines 11-21: proposal, test and refutation

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-01 | `11-12`, `their value depends on eliminating their own false explanations` | the two instances of `their` refer to different things, and the personification is heavy | `REWRITE`: take candidate statements as an explicit subject and express it as testing and eliminating false explanations | establishes that proposing is easy and eliminating error is what counts |
| R-02 | `13-15`, `experimental structures satisfy and damaged versions fail` | `damaged versions` does not say they come from the same parent | `REWRITE`: state at first mention that they are controlled perturbations of experimental parents | defines the relation between the positive and negative structures |
| R-03 | `15-17`, `proposed and implemented hypotheses, selected thresholds..., designed counterexamples...` | the construction and refutation stages are crowded into one sentence | `SPLIT`: the first sentence covers proposal/implementation/threshold; the second covers counterexamples/held-out/physical controls | advances along the discovery workflow |
| R-04 | `18-21`, `Refuted claims stayed ... making that search a sequence...` | the use of the failure record and the methodological conclusion share a sentence | `SPLIT`: first say the record becomes diagnostics for the next round; then summarise falsifiable experiments with `Thus` | explains the refutation loop clearly |

### The Figure 1 caption, lines 27-41

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-05 | `31-33`, t-SNE `projection of ... statements; blue intensity ... red circles...` | the data object, the colour and the red points are three encodings in one sentence | `SPLIT`: the first sentence gives the object of the two-dimensional projection; the second the blue density; the third the red circles for Law 1-8 | the panel reads as "object -> colour -> marker" |
| R-06 | `36-37`, `The eight laws Law 1--Law 8... Set 1, Set 1'...` | `eight laws Law 1--Law 8` repeats, and the list of sets drags | `CUT/REWRITE`: delete the repeated naming; keep the predicates, the set membership and the two performance figures | describes panel d concisely |
| R-07 | `38-41`, running best, the two curves, the inset, the strip | four kinds of visual information in one long sentence | `SPLIT`: one sentence for the main curve; one short sentence each for the inset and the outcome strip | lets panel e be decoded in order |

### Results 2.1, lines 45-56: the scale of the search and its results

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-08 | `45-48`, `fixed splits of ionic structures from 99,162 experimental ICSD and COD structures` | `structures` repeats, and the data sources are inserted into the main clause | `REWRITE`: put the total in the main clause and move ICSD/COD and the Methods/SI pointers to the end | states the analysed population quickly |
| R-09 | `50-51`, `Scale mattered because it multiplied the chances...` | `Scale mattered` is empty | `REWRITE`: state directly that repeated datasets and controls created more opportunities to refute passing claims | explains the scientific role of scale |
| R-10 | `51-53`, 572 investigations and 2,037,606 evaluations | clear, but two million appears again at line 56 | `KEEP` the scale sentence; line 56 carries only the closing "eight survived", so the same number is not fully explained twice | reports the scale once and the result once |
| R-11 | `53-55`, `failed controls through misleading metrics, class imbalance, recognition...` | four kinds of failure enumerated in succession, hard for a reader to structure | `SPLIT`: first say 11 conclusions failed; then organise the causes into three groups, metrics/data imbalance, perturbation recognition, and implementation | shows the outcome of refutation |
| R-12 | `55-56`, `Most investigations changed nothing: ... improved only four times...` | two layers of temporal information follow the colon | `SPLIT`: the first sentence says most investigations did not change the best set; the second reports the four improvements and the last one after 500 | foregrounds that discovery does not accumulate linearly |
| R-13 | `56`, `Of two million evaluations, eight one-line laws survived...` | a forceful paragraph conclusion | `KEEP`, de-duplicating only against R-10 | completes the search funnel |

### Results 2.1, lines 58-64: the law-set sequence

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-14 | `58-61`, Set 1-4, Set 1', and the figure and SI pointers in one sentence | the main sequence and the auxiliary set are crowded together | `SPLIT`: the first sentence defines only the strictness sequence Set 1-4; the second covers Set 1' and where the full definitions live | establishes the main sequence first, then the auxiliary choice |
| R-15 | `62-63`, `early screening may favour..., database auditing fuller...` | the second half elides its predicate and the parallel is incomplete | `REWRITE`: supply `whereas database auditing may favour fuller damage detection` | explains why different tasks choose different strictness |
| R-16 | `63-64`, `Choosing ... therefore requires two measurements` | clear | `KEEP` | leads into the next subsection |

### Results 2.2, lines 69-74: the two metrics and the baselines

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-17 | `69-70`, the two definitions | clear but too dense within the line | `SPLIT`: give satisfaction and damage detection a sentence each, in the same construction | defines the two core metrics in parallel |
| R-18 | `70-72`, `bound that every structure satisfies ... converse bound scores the reverse` | `the reverse` leaves the reader to work it out | `REWRITE/SPLIT`: state the two numerical directions of the loose and the strict bound explicitly, then conclude that both are needed | explains the trade-off intuitively |
| R-19 | `72-74`, the charge-neutrality and fixed-distance baselines | two failure modes in the same paragraph with no explicit contrast | `SPLIT`: one sentence for the composition-only baseline; one for the element-blind distance baseline | forms a map of the two baseline classes |

### Results 2.2, lines 76-87: Law 1 and the DFT energy scale

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-20 | `76-84`, the introduction of Law 1, the formula and the variable definitions | clearly structured | `KEEP`; only keep the definitions after the equation to one sentence | establishes the reduced contact |
| R-21 | `84-85`, an absolute cutoff is stricter for small ions | clear | `KEEP` | explains the scale normalisation |
| R-22 | `85-87`, the threshold definition and the performance in one sentence | the rule definition and the result are mixed | `SPLIT`: the first sentence defines 0.735; the second reports 99.2%/28.9% | separates the law from the evidence |
| R-23 | `87`, `To put ... we rigidly scaled twenty...` | the purpose, the sample and the DFT action can be separated | `SPLIT`: one sentence for the method; a new sentence for the result | states the verification design first |
| R-24 | `87`, `median 0.1 eV ... so both floors ... and ... 1.80 times... which is what makes...` | four layers nested: the energy crossing, the interpretation of the thresholds, and the cross-chemistry advantage | `SPLIT` into three sentences: the crossing; the energy region of the two floors; the 1.80-fold localisation and the significance of a shared coordinate | completes the physical calibration clearly |

### Results 2.2, lines 89-112: the eight formulas and the definitions

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-25 | `89`, `Seven further laws take the same form`, after which the formulas list Law 1 again | the lead-in does not quite match what is displayed | `REWRITE`: write `Together, the eight laws are`, or list only Law 2-8 in the body text | removes the momentary "seven vs eight" conflict |
| R-26 | `90-105`, the full eight formulas | high information density in the main text, though the formulas themselves have no linguistic problem | `KEEP the formula content`; when implementing, consider keeping only a representative expression and a Fig. 1d pointer, with the full set still in the SI of the same submission package | lowers the cognitive load of the main text without losing information |
| R-27 | `106-108`, Madelung, formal charge and BV run together, and `and BV bond valence` is grammatically incomplete | the three quantities are not fully defined | `SPLIT/REWRITE`: one sentence per quantity; give the full name of BV and the quantity it denotes | establishes the symbols one at a time |
| R-28 | `109-111`, the trigger condition and no verdict | two conditional rules in one sentence | `SPLIT`: the first sentence says an unmet trigger satisfies a conditional law; the second that a missing required input returns no verdict | distinguishes an untriggered condition from a missing input |
| R-29 | `111-112`, `Each law answers one question...` | clear | `KEEP` | motivates the combination |

### Results 2.2, lines 114-125: the performance progression of the combined laws

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-30 | `114-116`, expansion, exchange and same closest distance in one sentence | the list of three blind spots is a little heavy | `SPLIT`: one sentence for the overall judgement; the three examples as parallel short clauses or a compact list | establishes why complementary mechanisms are needed |
| R-31 | `117-118`, Set 3 performance | clear | `KEEP` | reports the intermediate law set |
| R-32 | `119-120`, `two and a half times ... one in twelve rather than...` | one sentence gives both the detection gain and the rejection cost | `SPLIT`: one sentence for the gain, one for the cost, keeping every number | makes the trade-off easy to scan |
| R-33 | `120-124`, the composition and performance of Set 4 | the rule composition and the results run together densely | `SPLIT`: one sentence adding Law 7/8; one for the overall and per-class performance | says why it is stricter before reporting the effect |
| R-34 | `124-125`, `gain came from combining laws...` | clear | `KEEP` | a local inference |

### Results 2.2, lines 127-133: comparison with existing criteria

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-35 | `127-130`, the distance-filter and Pauling comparison, and the sample conditions | the main comparison and the sample restriction are spread across two sentences | `REWRITE`: use a symmetric sentence for the two opposite failures first, then place the 5,297/charge-balanced conditions after the 6.5% result | foregrounds the core comparison |
| R-36 | `131-132`, `not greater strictness but readable chemical discrimination...` | a noun fragment after the colon | `REWRITE` into a complete causal sentence: PRIS occupies the useful region by combining readable chemical discrimination rather than uniform strictness | explains where the advantage comes from |
| R-37 | `132-133`, cutoffs fixed -> test verdict dependence | clear | `KEEP` | leads into the split check |

### The Figure 2 caption, lines 138-149

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-38 | `147-149`, re-derived constants, relative displacement and the right-hand flips in one sentence | the result points, the normalisation and the right-hand column crowd together | `SPLIT`: first the points at unity; then the right-hand verdict changes | lets panel d be read in order |
| R-39 | the other panel descriptions | clear | `KEEP` | unchanged |

### Results 2.2, lines 153-163: split consistency

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-40 | `153-157`, the three classes of cutoff, the electrostatic cutoffs, the verdict changes and Law 7 | several quantities stacked in succession | `SPLIT`: one sentence for the small threshold shifts; one for the electrostatic shifts; one for verdict stability | separates parameter change from output change |
| R-41 | `157-161`, five Supplementary Figures and their subjects listed in succession | the list of figure numbers interrupts the main argument | `COMPRESS/MOVE`: group the five into threshold, band and chemistry, and collect the references at the end of the paragraph | keeps the navigation without turning the text into a contents list |
| R-42 | `161-163`, `Verdicts transfer because ... while ...` | one sentence containing the mechanism, the direction, the domain, the population threshold and interpretability | `SPLIT`: the first sentence explains that the mechanism fixes what/direction/domain; the second that the population only sets the cutoff; the third gives the testability inference | completes the logical explanation clearly |

### Results 2.3, lines 168-177: an overview of the five mechanisms, and MgAl2O4

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-43 | `168-172`, screen useful, the law form, the eight laws and the five mechanisms | `mechanism` repeats across two sentences and the list is too long | `MERGE/SPLIT`: a topic sentence defining mechanism-resolved diagnosis, followed by the five mechanisms split into geometry/packing and electrostatics/bonding/symmetry | gives the reader a map of the mechanisms |
| R-44 | `173-174`, the five damages to MgAl2O4 and the inverse-spinel parenthesis | the case action and the background note share a sentence | `SPLIT`: the main clause states only the five damages; the inverse spinel becomes a short standalone note | establishes the case first, then adds the natural counterpart |
| R-45 | `174-176`, compression, expansion and exchange responses in succession | high information density | `SPLIT`: one sentence for the change in contact; one for the unchanged distance with like-charge bonds | shows why a single distance is not enough |
| R-46 | `176-177`, `so no single ... Each unsatisfied...` | the inference is clear | `KEEP`, optionally making the second sentence the end of the paragraph | closes on diagnosis |

### The Figure 3 caption, lines 182-196

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-47 | `190-196`, panel c | the curves, the twin axes, the thresholds, the domains, the 0.1 eV line and the SI pointers are all concentrated together | `SPLIT` into three sentences: the curves and the distribution; the law boundaries and 0.1 eV; where the per-compound and control results are in the SI | reads as data, thresholds, then extended evidence |
| R-48 | panels a/b | clear | `KEEP` | unchanged |

### Results 2.3, lines 200-223: the mechanistic explanation

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-49 | `200-211`, Law 1 -> Law 2 -> Law 3 | a clear progression, each sentence with a single function | `KEEP` | one of the strongest mechanistic paragraphs |
| R-50 | `213-215`, `Packing bounds still cannot ... so Law 4-6 move... and distinguish...` | the blind spot, the change of layer and the three failures share a sentence | `SPLIT`: first the fixed-coordinate exchange blind spot; then the introduction of the electrostatic laws | moves from geometry into electrostatics |
| R-51 | `215-216`, `Law 4 limits ... and Law 5 the largest...` | Law 5 elides its predicate | `REWRITE/SPLIT`: a complete sentence for each law, explaining the range and the maximum separately | distinguishes the two electrostatic diagnoses |
| R-52 | `216-217`, the Law 6 domain and fixed-distance exchange | clear, but could parallel the previous sentence | `REWRITE`: take Law 6 as an explicit subject and express the like-charge condition in full | completes the three electrostatic laws |
| R-53 | `217-223`, the Ewald limitation, Law 8, the distance response and the median values | four layers in succession | `BREAK`: one paragraph for the motivation for Law 8 and the physical response; a closing sentence for the experimental and damaged medians | moves from mechanistic explanation into quantitative evidence |

### Results 2.3, lines 225-233: Law 7 and the applied questions

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-54 | `225-229`, the definition of Law 7, the damage action and the Pauling relation | three propositions densely packed into two sentences | `SPLIT`: define the threshold; state the effect of perturbation and element reassignment; then give the empirical chemical-order interpretation | establishes the site-complexity mechanism |
| R-55 | `229-230`, `Because ... detection gaps, damage detection accumulates...` | nominalised, with the main clause at the end | `REWRITE`: take adding complementary mechanisms as the subject and say that the detection gaps are filled level by level | explains the accumulation in Fig. 3b |
| R-56 | `230-233`, the diagnosis list and the rhetorical question | logically clear | `KEEP`; break before the rhetorical question if needed | leads towards the applications subsection |

### Results 2.4, lines 238-251: entering the validation queue

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-57 | `238-239`, `The two criteria in use there ... same benchmark shows both` | `there/two criteria/both` can only be decoded at the end of the sentence | `REWRITE`: name the Pauling rules and the distance floor at the start of the sentence and remove the vague back-references | establishes the two baselines immediately |
| R-58 | `240-243`, `We benchmarked ... cutoffs ..., which detected...` | `which` may modify pipelines or cutoffs | `SPLIT`: one sentence for the comparison setup; then report the 1.6-3.2% with the two distance cutoffs as the subject | separates method from result |
| R-59 | `243-247`, Set 4, the matched cutoff, chemical selectivity and the positioning | four conclusions in succession | `SPLIT`: one sentence each for Set 4 performance, the matched comparison and the chemical-selectivity inference | builds a complete but readable chain of comparison |
| R-60 | `247-250`, the aggregate concern and the class-resolved/withheld classes | two kinds of validation placed after the colon | `SPLIT`: raise single-class dominance first; then write the selected classes and the withheld classes separately | rules out domination by a single perturbation |
| R-61 | `250-251`, `The same evidence can also be asked to screen...` | an awkward English collocation | `REWRITE`: put it directly as `We next asked whether...` | transitions to synthesizability |

### Results 2.4, lines 255-276: the definition of PSS

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-62 | `255-258`, the plausibility barrier, the continuous companion, the fitted pairs | the two sentences are logically clear but the terminology is concentrated | `REWRITE`: the first sentence states the need for a move from discrete PRIS to continuous PSS; the second simplifies the recorded/computed-only pairs to same-composition pairs | establishes the motivation for PSS and what it is fitted on |
| R-63 | `269-271`, the text lists five classes but the formula has six terms | a reader may think one has been left out | `REWRITE`: state six terms explicitly, and say that connectivity supplies both `kmax` and `fiso` | makes the text and the formula agree term by term |
| R-64 | `271-274`, the dominant coefficient, the random halves, the ranking/sign, the correlation | the properties and the robustness checks are crowded together | `SPLIT`: one sentence for the dominant volume effect; one for coefficient resampling; one for term correlation | separates the meaning of the score from its robustness |
| R-65 | `275-276`, `stronger expression ... and testing it requires...` | `it` may be the score or the interpretation | `SPLIT/REWRITE`: define a higher PSS first; then say explicitly `Testing this interpretation requires...` | transitions smoothly into the PU proxy |

### Results 2.4, lines 278-308: the PU proxy and the linear relation

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-66 | `278-282`, the database difficulty, CLscore, the training pool | the literature background and the present training setup are joined too quickly | `SPLIT`: one sentence for the scarce labels; one for the PU/CLscore principle; one for the present pool | moves from the problem to the solution |
| R-67 | `282-286`, the full architecture of both scorers in one sentence | a heavy cognitive load for a non-ML reader | `SPLIT`: a sentence each for CGCNN-PU and the frozen-MatterSim PU, keeping learned representation and inherited representation parallel | explains the two different routes |
| R-68 | `286-289`, `Two such different routes ... make ... unlikely... The 364,592...` | the abstract inference and the definition of the proxy set are awkwardly joined | `REWRITE/SPLIT`: first say that agreement reduces a model-specific artefact; then define the consensus-low set | arrives at the hard-negative proxy |
| R-69 | `291-294`, the matched-satisfaction comparison of Set 4 and PSS | three percentages in one sentence | `SPLIT`: one sentence for Set 4; open the next with `At the same satisfaction` for PSS | directly comparable |
| R-70 | `294-296`, the performance of the hull threshold and the extra computation it needs | the result and the cost share a sentence | `SPLIT`: report the 72.0% first; then the difference between requiring a relaxation and a phase hull, and a direct screen | compares effect and route |
| R-71 | `296-298`, `agreement can manufacture a trend ... each trend below...` | figurative, and `each trend below` is vague | `REWRITE`: name the shared-selection risk directly, and say explicitly that CLscore-Set 4 and CLscore-PSS are checked separately below | explains the purpose of the per-model check |
| R-72 | `300-303`, both models described in full again | repeats 282-286 | `CUT/MERGE`: change to `Both PU models`, with the method and figure references collected at the end of the sentence | focuses on the shared direction of the results |
| R-73 | `303-305`, two trends, the endpoints and two R2 values | the reader has to reread to map the numbers | `SPLIT`: a sentence each for the Set 4 trend and the PSS trend, with each R2 immediately after its own analysis | makes the mapping of the numbers clear |
| R-74 | `306-308`, the population relation -> the individual polymorph test | the reasoning is clear | `KEEP`, optionally making the second sentence the start of a new paragraph | leads into ranking |

### The Figure 4 caption, lines 313-337

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-75 | `313-314`, the figure title covers damage, synthesizability and inverse design at once | three independent tasks side by side | `REWRITE`: put physicochemical screening at the head and drop the details the panels can carry | gives Figure 4 a single question |
| R-76 | `319-322`, panel c | the axes, the four groups of laws, the three classes of curve and the connectors in one sentence | `SPLIT`: one sentence for the axes and curves; one for the matched connectors | identify the content first, then the key comparison |
| R-77 | `323-326`, panel d | the series, the shading and the straight-line R2 in one sentence | `SPLIT` into three sentences: curves, range, fit | reads in order |
| R-78 | `330-337`, panel f | the main plot, the Set 4 point, the inset, the DFT dashed curve and the scale conversion | `SPLIT` into three sentences: the threshold sweep; the inset descriptors; the DFT verification | separates the main analysis, the explanation and the verification |
| R-79 | panels a/b/e | clear | `KEEP` | unchanged |

### Results 2.4, lines 341-355: same-composition ranking

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-80 | `341-345`, the task difference, the Set 4 ties, active refutation | the first two sentences are clear; the third stands as isolated metadiscourse | `MERGE`: combine `Active refutation was needed...` with the specific tie-counting error | goes straight to the error and the correction |
| R-81 | `345-350`, the tie error, the symmetry explanation, the audit, the space-group medians, the tie correction | one case takes six sentences and interrupts the ranking storyline | `COMPRESS` into three steps: the mis-scoring; the composition-controlled refutation; ties treated as no choice. The detailed medians go to the SI | keeps the discovery process but shortens the digression |
| R-82 | `347`, `A ... audit refuted it` | `it` is unclear | `REWRITE`: say explicitly that it refuted the symmetry-based explanation | avoids being read as refuting the data |
| R-83 | `350-355`, the overall accuracy, the top fifth, the PSS/DFT division and the transition to inverse design | the results and the entry to the next experiment share a sentence | `SPLIT`: a sentence each for overall, high-confidence, the division of labour, and the next experiment | moves naturally from the ranking result to inverse design |

### Results 2.4, lines 357-364: the main inverse-design result

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-84 | `357-360`, the MatterGen runs, the 1,081 candidates, the definition of UMA and the 140 target candidates | the task setup, the model identity and the result run together | `SPLIT`: a sentence each for the generation setup, the role of UMA and the 140 result | advances as task -> proxy -> subset |
| R-85 | `360-362`, `These structures supplied only...` | `These structures` may be the 1,081 or the 140 | `REWRITE`: say `the 1,081 generated candidates` explicitly, then state the two PSS terms and the threshold calibration | fixes the input set clearly |
| R-86 | `362-364`, 61 removed, all 140 retained, a 5.6% reduction | three equivalent or related numbers in one sentence | `SPLIT`: one sentence for the screening count and reduction; one for the retention | foregrounds efficiency and retention |
| R-87 | `364`, the first-principles moduli, the proxy ratio, one candidate, the ranking and the 99.2% all on the same source line | one physical verification carrying four propositions | `BREAK/SPLIT`: a sentence each for the DFT campaign, the proxy-DFT scale, the pair ranking and the 99.2% retention | establishes a separate DFT verification paragraph |
| R-88 | `364`, `The proxy ran high by a median factor of 0.940` | `ran high` appears to contradict a factor below 1 | `REWRITE`: state the numerator and denominator of the ratio explicitly, then say the proxy runs high | removes the ambiguity of direction |
| R-89 | `364`, `Relaxation also revised what the screen had been reading` | personified, and it switches abruptly to Law 7 | `BREAK/REWRITE`: a new paragraph stating directly that DFT relaxation changed the Law 7 classifications | moves from property verification into structural change |
| R-90 | `364`, 61 -> 113, `not one moved the other way` | two time points and the direction in one sentence | `SPLIT`: three short sentences for before, after and direction | shows the change clearly |
| R-91 | `364`, `The gain sat in...` | an unnatural collocation, and `gain` is vague | `REWRITE`: state explicitly that all additional Law 7 passes occurred among PSS-retained candidates | identifies where the change came from |

### Results 2.4, lines 366-379: the screened structures and tunability

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-92 | `366`, `what the 61 share` | referring back by number alone, which conflicts with several earlier 61s | `REWRITE`: write `the 61 PSS-screened candidates` | pins down the set |
| R-93 | `367-371`, Law 7/open packing and the four metrics of the two Ir2Os7 structures | many example values, but the logic is clear | `REWRITE` into strictly parallel sentences: the symmetry/volume/PSS of the screened candidate; the same sequence for the retained one | makes the pair comparison easy to scan |
| R-94 | `373-375`, `they span two extremes` | `they/extremes` are abstract | `REWRITE`: say explicitly that one operating point barely shortens the queue and the other shortens it strongly | frames the Set 1-3/Set 4 numbers |
| R-95 | `375-377`, the definition of the PSS subset and the 140/140 retention | the condition and the result are nested | `SPLIT`: define first which Set 4 violations PSS screens; then report that all the high-property candidates are retained | explains what the continuous score does |
| R-96 | `377-378`, `reduced ... by up to 67.3%` | it sits a long way from the earlier 5.6%, and the reader may not see that it comes from tunable operating points | `REWRITE`: keep the strong 67.3% conclusion while making clear that it is the largest queue reduction across tunable PRIS/PSS operating points | connects the tunability to the headline result |
| R-97 | `378-379`, `PRIS states why ... PSS sets how strongly...` | concise and parallel | `KEEP` | closes the PRIS/PSS division of labour |

### Results 2.5, lines 386-398: from the queue to the database problem

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-98 | `386-391`, the energy-phonon classes, the recorded/unrecorded percentages, and distance/relaxation/site splitting | about 60 words with four layers of nested information | `SPLIT`: one sentence for the percentage observation; one explaining that distance and relaxation still miss site splitting | establishes the incremental value of Law 7 |
| R-99 | `391-395`, the GNoME and A-Lab controversy | the Introduction has already given the history in full | `COMPRESS` to one sentence of background, keeping only why site ordering is the key question, plus the citations | keeps Results from repeating the Introduction |
| R-100 | `395-398`, `remain enriched ... controversy turns on why ... diagnosis...` | `controversy` repeats and `turns on why` is vague | `CUT/REWRITE`: delete the abstract closing phrase and state directly, with Law 7 as the subject, that it measures the site splitting geometric screens miss | moves quickly to a testable mechanism |

### The Figure 5 caption, lines 403-429

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-101 | `404-408`, panel a | the axes, the control, the markers and the offset, four kinds of note | `SPLIT`: one sentence for the data object and axes; one for the marker encoding; one for the offset | reads in order |
| R-102 | `409-413`, panel b | the Upper and Lower subpanels and three marker classes in one sentence | `SPLIT`: one sentence for Upper; one for Lower; the median and 300 K markers explained with Lower | lets the two subpanels be decoded independently |
| R-103 | `414-419`, panel c | the MLIP distributions, the DFT curve, and why no damaged DFT is drawn, three layers | `SPLIT`: the main distributions; the DFT parent curve; the population note | separates the figure content from the sample note |
| R-104 | `420-423`, panel d | the curves, the medians, the damage references and the axis range in one sentence | `SPLIT` into three sentences: data, reference, axis | lightens the caption |
| R-105 | `427-429`, panel f | the per-structure CPU and the queue total in one sentence | `SPLIT`: the lower axis for a single structure; the upper axis for the whole queue | distinguishes the two scales |
| R-106 | panel e | clear | `KEEP` | unchanged |

### Results 2.5, lines 433-451: the seven generators and GNoME

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-107 | `433-435`, the seven-generator sample | clear | `KEEP` | states the sample |
| R-108 | `436-440`, the distance result, the symmetry dependence of Set 4, and three satisfaction figures | the summary and the numbers are too dense in one sentence | `SPLIT`: one sentence for the distance baseline; one for the symmetry dependence; one for the three figures | baseline first, then the main phenomenon |
| R-109 | `440-445`, Law 7, relaxation, overlap, the Wyckoff construction and the design choice | the evidence and the design inference are completed within one paragraph and the sentences run too long | `SPLIT`: the Law 7 evidence; overlap vs symmetry; the shared generator construction; the design inference | advances from observation to a design principle |
| R-110 | `445-451`, `The same failure ... GNoME ... sample ... rates ... strain alternative` | no paragraph boundary between the generators and GNoME | `BREAK` at 446; the first sentence states `We next tested this pattern in GNoME`; then a sentence each for the sample, the result and the alternative | forms a separate GNoME evidence unit |

### Results 2.5, lines 453-461: relaxation energy rules out gross strain

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-111 | `453-455`, the MatterSim 150 parents and three classes of median | clear, though the result sentence runs slightly long | `SPLIT`: the parent release; the compression and displacement medians | establishes the severity scale |
| R-112 | `455`, `To confirm ... 200 identical cells: ... 0.953, and twenty ... 0.0001` | four layers: purpose, sample, correlation and absolute magnitude | `SPLIT` into three sentences: the purpose and sample of the check, the rank correlation, and the parent median | states the why, the agreement and the scale |
| R-113 | `455-458`, the MatterGen question and the `Yet` result | the answer is delayed | `REWRITE`: write `The answer was no` directly, then give the 0.007/0.006 | answers immediately whether a small energy is sufficient |
| R-114 | `459-461`, the gross-strain inference -> the merge hypothesis | logically correct but it changes question rather quickly | `SPLIT`: the first sentence closes gross strain; the second states that site splitting remains; the third proposes the merge test | moves naturally into the label intervention |

### Results 2.5, lines 463-472: the merge intervention and DFT ordering

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-115 | `463-466`, 113/150, 78%, 79%, and 27 controls | two results and the control referred back to as `latter` | `SPLIT`: a sentence each for mergeable prevalence, bound restoration, the space-group increase and the experimental control | shows the outcome of the intervention layer by layer |
| R-116 | `467-469`, the sample conclusion, the controversy, the label-only change and `thermodynamically empty` | the conclusion and the next DFT experiment are crowded into one group of sentences | `BREAK`: close the merge with one strong sentence; then open a new paragraph for the DFT ordering | separates the structural intervention from the thermodynamic verification |
| R-117 | `469`, 23 entries/10 controls, 0.0001/0.036, and 18/23/0 controls | the method, the energies and the temperature conclusion in one sentence | `SPLIT`: three sentences for the enumeration method, the energy comparison and the 300 K counts | builds a clear thermodynamic chain of evidence |
| R-118 | `470-472`, the critique -> the measured mechanism -> the `converse error` | the conclusion and the transition to the next topic share a sentence | `SPLIT`: the first sentence closes ordering; move the complementary identity error to become the topic sentence of the next paragraph | moves from ordering into the wrong element |

### Results 2.5, lines 474-492: element identity at fixed coordinates

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-119 | `474-478`, the Harrison history and how Spek found it | the historical detail of the second sentence is long | `COMPRESS`: keep the background needed for the genuine diffraction, the changed identity and the coordinate checks; reduce the discovery process to a subordinate clause | establishes the experimental precedent quickly |
| R-120 | `479-482`, reports may alter cells -> fixed-coordinate exchange -> three quantities -> the conclusion | the experimental design and the mechanistic result run together | `SPLIT`: why fixed coordinates; which physical quantities change; why a coordinate screen misses it | establishes the controlled identity test |
| R-121 | `484-488`, six PRIS against four check families, and two classes of numerator | the comparison setup is clear but the numeric sentence is dense | `SPLIT`: a sentence each for the comparison design, the cation-cation case and the cation-anion case | reports the two exchange classes in parallel |
| R-122 | `489`, `A recovered archive ... agrees` | jumps abruptly from the benchmark to a historical archive | `BREAK`: make this sentence the start of a new paragraph, rewritten as `The same mechanisms recur...` | separates the controlled test from the archive case |
| R-123 | `489-492`, four framework entries, two metrics and a fifth entry | the sample and the two law values are too dense | `SPLIT`: a sentence each for the four related entries, the Law 1 result, the Law 8 result and the fifth entry | maps the archive onto the laws one item at a time |

### Results 2.5, lines 494-509: speed and the hierarchy

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-124 | `494-497`, Law 1, all eight laws and the DFT queue time | three time scales in succession | `SPLIT`: a sentence each for Law 1, the eight-law implementation and the DFT estimate | forms a cost gradient |
| R-125 | `497-500`, the millionth of the cost, the three application settings and the mechanism review | one conclusion sentence dragged out | `SPLIT`: a sentence each for the relative cost, where it is used and its diagnostic value | moves from speed to significance in application |
| R-126 | `502-506`, the thermodynamic/dynamic/synthesis distinction and three sets of statistics | three sets and their statistics in one sentence | `SPLIT`: a sentence each for on-hull imaginary modes, recorded metastability and the 4,271 unrecorded stable entries | separates the three kinds of evidence |
| R-127 | `507-509`, plausibility precedes them | the core hierarchy sentence is clear | `KEEP` | closes the Results |

### Discussion, lines 514-526: the three decision scales

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-128 | `514-515`, `three levels ... entry to queue down to element on site` | the framing sentence omits the database level and its spatial direction is convoluted | `REWRITE`: list the validation queue, the database and the individual site in parallel directly | gives a clear map of the three layers that follow |
| R-129 | `515-522`, three instances of `At the ... level` | the parallel structure is good, but each sentence is slightly long | `KEEP the three-sentence structure`; each sentence keeps one mechanism and one implication, dropping explanations Results already reports in full | completes the cross-result synthesis |
| R-130 | `522-526`, the independent layer, what it asks, that it precedes the three checks, and that they cannot collapse | four layers of summary crowded into two sentences | `SPLIT`: establish the layer; what it asks; why it precedes; why the scores differ | gives the conceptual conclusion step by step |

### Discussion, lines 528-544: the PRIS/PSS division of labour

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-131 | `528`, `The first of these checks` | `first` may refer to the queue or to plausibility | `REWRITE`: write the structural-plausibility check directly | makes the subject explicit |
| R-132 | `528-531`, diagnosis, strictness and the five mechanisms | the two sentences are clear | `KEEP`, removing only the repeated `mechanism` | explains what PRIS outputs |
| R-133 | `533-536`, the PU link, the continuous PSS score and the five-term list | Results already defines the score, and it is enumerated again in full here | `COMPRESS`: keep the unexpected link and the continuous tunable score; drop the full descriptor list | the Discussion synthesises rather than redefines |
| R-134 | `536-537`, the inverse-design dense packing and Law 7 detail | repeats a local mechanism from Results | `COMPRESS/MERGE` into the PRIS/PSS division of labour, with one sentence saying PSS weighs competing mechanisms | forms a cross-result synthesis |
| R-135 | `537-542` and `539-542`, PSS sets the threshold and PRIS names the mechanism, stated twice | adjacent repetition | `MERGE`: keep only the fullest single statement, `PRIS bounds/names; PSS orders`, adding the measurable vocabulary and the lower-cost outcome | states the roles and the value once |
| R-136 | `542-544`, the database/training labels and no verdict | switches abruptly from inverse design to applications | `BREAK` and add `Beyond queue screening`; one sentence for the database and training uses, one for missing-input handling | adds the deployment uses |

### Discussion, lines 546-556: closing on active refutation

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| R-137 | `546`, `Confidence ... rests ... because active refutation...` | two nested layers of causation | `SPLIT`: say first that confidence rests on the retention process; then take active refutation as the subject | enters the methodological significance |
| R-138 | `546-551`, 572, 2,037,606, 11 and 8 all given again in full | repeats the search-scale paragraph of Results | `COMPRESS` to a one-sentence anchor on the scale, keeping every strong number but not explaining the whole process again | the Discussion does not re-enact Results |
| R-139 | `551-553`, the failure record, the call in the literature, `This record supplies it` | `it` refers abstractly | `REWRITE/MERGE`: say explicitly that the preserved record provides systematic reporting of failed attempts, with the citation next to the call | connects the record to the need in the field |
| R-140 | `553-554`, `no longer asks whether ... but explains why` | `asks whether/explains why` are not parallel | `REWRITE`: `does not stop at a pass/fail verdict; it explains why...` | a strong and clear diagnostic conclusion |
| R-141 | `554-556`, the closing autonomous-agents sentence | the strong conclusion is clear | `KEEP`; check only the position of the parenthesis and the sentence length | serves as the last sentence of the paper |

## A5. `methods.tex`: Methods

### Opening paragraph, lines 3-7

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-01 | `3-7`, `The analysis coupled ... The subsections below...` | each sentence has a clear function | `KEEP` | states the overall design and the Methods/SI division |

### Data sets and study design, line 11

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-02 | `11`, `Threshold fitting and held-out assessment used splits assigned before any evaluation.` | this is the key design principle, but the more specific written protocols appear only afterwards, inverting the order | `MOVE/MERGE`: open the paragraph with the written protocols fixing the split, criteria and allowed quantities; then say the split was assigned before evaluation | gives the pre-fixed principle first and the execution afterwards |
| ME-03 | `11`, `We analysed 99,162 experimental structures...` | clear | `KEEP` | establishes the data population |
| ME-04 | `11`, `For law discovery, a seeded hash ... discovery, held-out or reserve split.` | clear, but it needs to sit next to the split principle | `MOVE` to just after ME-02, keeping the seeded hash and the three splits | says how the pre-assignment is implemented |
| ME-05 | `11`, `Eligibility ... ionic structures, and splitting divided structures rather than compositions.` | eligibility and the split unit are two independent design decisions | `SPLIT`: one sentence for ionic eligibility; one for the structure-level split | keeps the reader from taking them for the same restriction |
| ME-06 | `11`, `Every damaged structure ... inherited its parent's assignment.` | the key anti-leakage rule is clear | `KEEP` and place it immediately after the split-unit sentence | completes the parent-child split logic |
| ME-07 | `11`, `Written protocols fixed ... before evaluation.` | placed too late and repeats the opening | per ME-02, `MOVE` to the head of the paragraph and merge with the first sentence | declares the protocol lock first |
| ME-08 | `11`, `Thresholds were fitted on 12,632 ... then assessed on 5,297...` | four sample sizes in one sentence remain readable, but discovery and held-out should be visually separated | `SPLIT`: one sentence for the discovery sample; one for the held-out sample | pairs the fitting and the assessment one to one |
| ME-09 | `11`, `No threshold was fitted ... although those results informed...` | the distinction between "not fitted" and "influenced advancement" matters, and `although` is easily skimmed past | `SPLIT`: the first sentence keeps no threshold fitted; the second says explicitly that the held-out results were used only for advancement decisions | separates parameter fitting from model-set selection clearly |
| ME-10 | `11`, `After law-set selection, a split-labelling error ... so the reserve...` | the error, the action and the consequence are compressed into one sentence | `SPLIT`: first say that reserve rows entered one full-sample fit; then that the reserve is therefore no longer an independent final test | states the incident and its consequence directly |
| ME-11 | `11`, `Supplementary Note ... reports the error ... Database inventories...` | the two SI pointers serve different purposes | `BREAK`: keep the incident citation at the end of the split paragraph; move the inventory and licence citation to the next short sentence or the end of the paragraph | keeps the navigation from interrupting the main design chain |

### Structural descriptors and law-set evaluation, lines 15-19

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-12 | `15`, `On these data, each Law 1--Law 8 quantity ...` | clear | `KEEP` | states the principle that each law is evaluable |
| ME-13 | `15`, the formal-oxidation-state and external-fallback sentences | the rules for the primary and the external analyses are clear but could be contrasted explicitly | `MERGE/REWRITE` into two parallel short sentences, `In the primary analysis...; in the external analyses...` | the two charge conventions are visible at a glance |
| ME-14 | `15`, `Native CIF oxidation annotations ... were excluded ... so that...` | the exclusion and the reason it avoids circularity share a rather long sentence | `SPLIT`: the first sentence lists the exclusion; the second says it avoids charge assignment and bond-valence evaluation sharing a bond length | separates the rule from its rationale |
| ME-15 | `15`, `Primary neighbours came from CrystalNN ..., and Law 7 used spglib...` | two unrelated algorithmic conventions joined by `and` | `SPLIT`: one sentence for neighbour assignment; one for the symmetry convention | separates coordination from symmetry |
| ME-16 | `15`, `On a separate 3,000-structure sample...` | this is a robustness result, and inserting it into a definitions paragraph interrupts the methodological chain | `MOVE` to the end of this subsection or just before the corresponding SI pointer, as a standalone robustness-check sentence | define first, then report the sensitivity of the method |
| ME-17 | `15`, `Discovery used deposited cells, whereas ... primitive cells, so...` | the two cell conventions and the restriction on comparison share a sentence | `SPLIT`: the first sentence gives the discovery and external conventions; the second says rates involving Law 7 are compared only under a shared convention | separates the rule from its statistical consequence |
| ME-18 | `17`, `Each law combines ...` | clear | `KEEP` | enters the definitions of the eight laws |
| ME-19 | `17`, the five sentences of Law 1: the shared coordinate, the radius sum, the repulsion, the cutoffs and the perturbation response | the order of information is essentially right, but the paragraph is too dense | `BREAK`: first the normalized contact and the meaning of the radius sum; then the physical direction and the cutoffs; finally the perturbation response | moves from definition to physical basis to operational response |
| ME-20 | `17`, `The ionic-character conditions exclude ... Law 2 and Law 6 ... so Law 2 ... and Law 6...` | the domain restriction of two laws and two thresholds/conditions are packed into one sentence | `SPLIT`: one sentence for the shared ionic-domain principle; one for the Law 2 condition; one for the Law 6 domain, with the Law 6 detail moved next to its exchange mechanism | avoids cross-referencing between Law 2 and Law 6 |
| ME-21 | `17`, `The coordination condition of Law 3 ... because...` | clear | `KEEP` | defines the domain of Law 3 and the reason for it |
| ME-22 | `17`, the two sentences for Law 4 and Law 5 | both concepts are readable, but they should be strictly parallel | `REWRITE`: give both sentences the same grammar, "quantity/contrast -> what it detects", each keeping one mechanism | makes global spread and worst-site instability easy to compare |
| ME-23 | `17`, `Both respond ... A cation--anion exchange...` | the first mention says wrong-site exchange and only the next sentence gives the cation-anion mechanism | `MERGE/REWRITE`: state coordinate-preserving exchanges first; then use the cation-anion case to explain why the distance matrix is unchanged while the electrostatics change | puts the phenomenon and its mechanism next to each other |
| ME-24 | `17`, `Law 8 places ... ceiling at 0.7143 and thereby...` | clear, but crowded into the same paragraph as the Law 4/5 exchange mechanism | `BREAK`: Law 8 becomes its own one-sentence paragraph, with the coverage citation after it | ends the law-definition sequence |
| ME-25 | `19`, `The main benchmark rates ... and applications expose ... no-verdict outcome.` | the benchmark denominator and the deployment outcome are two different things | `SPLIT`: one sentence saying the benchmark evaluates only available inputs; one saying applications report unavailable inputs as no verdict | separates the denominator convention from the operational label |
| ME-26 | `19`, `When required charges or radii are absent ... and a structure that violates...` | missing input and violation aggregation are joined by `and` but are logically different | `SPLIT`: missing inputs -> no verdict for the affected ionic laws; any evaluable violation -> implausible | defines two independent decision rules |
| ME-27 | `19`, the SI pointer sentence | clear | `KEEP` | points at the full definitions and lookup rules |

### Controlled damage and law selection, lines 23-25

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-28 | `23`, `The damaged structures came from five perturbations...` | the list of five is clear | `KEEP` | defines the damage families |
| ME-29 | `23`, `Independent contact, electrostatic and MatterSim-relaxation controls...` | clear | `KEEP` | describes the severity controls |
| ME-30 | `23`, `For each mechanism ... Law selection maximised...` | jumps straight from the mechanism semantics to the optimisation objective | `BREAK`: the mechanism-interpretability sentence closes the damage design; the selection objective starts a new paragraph | separates damage construction from law selection |
| ME-31 | `23`, `Fixed Set 4 was then evaluated once ... and compared with...` | one sentence contains both the one-shot evaluation and the optimal-tree comparator | `SPLIT`: one sentence for the held-out Set 4 evaluation; one for the depth-three tree comparison | separates the evaluation target from the comparator |
| ME-32 | `23`, `Re-derived at the held-out percentiles, Law 4 moved ... and Law 5 ... and these shifts altered...` | two threshold changes and two verdict impacts are too dense | `SPLIT`: three sentences for the Law 4 change, the Law 5 change and the two verdict impacts | lets the reader map each law to its impact |
| ME-33 | `23`, `For leave-one-damage-class-out ...` | a new validation protocol buried in a long paragraph | `BREAK` here; first say the tree and single threshold are fully repeated | marks the entry into the transfer test |
| ME-34 | `23`, `For Set 4, only its additions ... exposed to all five classes.` | the key restriction is clear but the sentence is long | `SPLIT`: one sentence for the fixed Set 3 base; one for only the additions being reselected | states the frozen/reselected boundary of the LOFO |
| ME-35 | `23`, `That variant therefore tests ... rather than...` | clear | `KEEP` | defines precisely what the test means |
| ME-36 | `23`, the LOFO 51% and 68.6-100% sentences | the two results concern different objects | `KEEP both sentences`, naming the Set 4 additions explicitly in the second | reports the transfer of single laws and of the additions |
| ME-37 | `25`, the sample, results and SI citations of the distance-cutoff benchmark | essentially clear; the three figures should stay parallel | `KEEP the paragraph structure`; make the subject of 99.1%/26.8% and 33.9% uniformly the method or the cutoff | makes the benchmark outcomes directly comparable |

### External evaluation, PSS and statistics, lines 29-37

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-38 | `29`, `With PRIS thresholds fixed, external evaluation covered...` | six external analyses listed in succession | `SPLIT` into "generated/computed structures" and "coordinate/history/relaxation diagnostics", keeping the fixed thresholds in the first sentence | gives the external validation a classification |
| ME-39 | `29`, the generator-cutoff and Law 7 rate sentences | the method baseline and the main result are clear | `KEEP`, but state the same relaxed benchmark and control convention at the start of the second sentence | prevents different denominators being misread |
| ME-40 | `29`, the GNoME sampling protocol and the (Pm/Cm) rates | cuts straight from the seven generators to GNoME | `BREAK`, opening a new paragraph with `We next sampled GNoME...`; a sentence each for the protocol and the result | forms a separate GNoME evidence block |
| ME-41 | `29`, the MatterGen relaxation-energy comparison | the sample and the three verdict classes share a sentence | `SPLIT`: one sentence for the purpose and sample of the comparison; one for the 134/11/355 outcomes | separates the analysis from the classification result |
| ME-42 | `29`, the 26,600 Materials Project structures sentence | no explicit relation to the previous sentence | `BREAK/REWRITE`: state that this cohort serves the thermodynamic-dynamical-record comparison | gives the number a task as its subject |
| ME-43 | `31`, the two same-composition ranking sentences | clearly expressed | `KEEP`; only keep "selected no unique structure" and the pair-level distinction as two separate metrics | avoids conflating composition level with pair level |
| ME-44 | `33`, `PSS was fitted ... as an antisymmetric, zero-intercept logistic score...` | two statistical terms opaque to a cross-disciplinary reader | `SPLIT`: give the model form first; then one plain sentence saying that swapping the pair order reverses the score and that equal structures have zero offset | explains why antisymmetric and zero-intercept |
| ME-45 | `33`, `Each development pair compared...` | clear | `KEEP` | defines the training pair |
| ME-46 | `33`, the five sentences defining the six descriptors | the definitions are complete, but the noun density within one paragraph is too high | `BREAK` into packing/electrostatics/bond valence and site/connectivity groups; each sentence gives the plain function before the symbol | lets a reader outside the field grasp the meaning before the formula |
| ME-47 | `33`, `Descriptors are standardised ... and unavailable descriptors ... medians.` | standardisation and imputation are two separate operations | `SPLIT` into two sentences | makes the preprocessing steps explicit |
| ME-48 | `33`, `The score was then frozen ... before transfer...` | clear | `KEEP` and place it at the end of the paragraph | completes the fit -> freeze -> evaluate -> transfer order |
| ME-49 | `35`, `Building on ... expanded ...` and `In previous work...` | the present work comes first and the earlier work second, inverting the timeline | `MOVE` the prior-work sentence to the head of the paragraph, with the current expansion next | lineage first, then the current study |
| ME-50 | `35`, `Consensus between a retrained 50-bag ... and 50 MLP ... defined...` | two model architectures, the bag count and the consensus outcome in one sentence | `SPLIT`: one sentence for CGCNN-PU; one for MatterSim-MLP-PU; one for the consensus tail | separates the models from the selection rule |
| ME-51 | `35`, the deduplication and AUC sentences | the selection result and the validation performance are mixed in one paragraph | `BREAK` at validation; give the 364,592 unique structures first, then the AUC of each model | moves from proxy construction into model validation |
| ME-52 | `35`, `At matched ... PSS screened 31.8 ... Trends were aligned ... and a balanced comparison...` | the PSS result, the percentile alignment and the hull comparator run together | `SPLIT`: a sentence each for the matched-satisfaction result, the alignment method and the balanced comparator | conclusion first, then how the comparison was made fair |
| ME-53 | `35`, `In that comparison, the hull threshold retained 86.2%...` | clear | `KEEP` | completes the comparator result |
| ME-54 | `37`, the MatterGen generation/UMA sentence | the two stages are clear but the sentence is long | `SPLIT`: the 13-run generation and deduplication; the UMA property proxy | separates generation from independent property assessment |
| ME-55 | `37`, `A set of 541 ... fixed threshold, retaining 528 ... before application to 1,081...` | three layers: calibration, retention and application | `SPLIT`: the calibration cohort; the 528 retained; the threshold applied to the generated set | separates threshold setting from deployment clearly |
| ME-56 | `37`, the frozen-medians sentence | clear | `KEEP` | explains the handling of missing terms |
| ME-57 | `37`, the 61 removed against the retained volume | the contrast is clear | `KEEP` | reports the direction of the screening |
| ME-58 | `37`, Set 4 45/140 and the PSS 61-of-728/95 retention sentences | the strict-set result and the tunable-screen result are crowded together | `BREAK`: the Set 4 outcome first; then the PSS-screened subset; then a single sentence foregrounding the retention of all 95 | shows the different roles of hard rules and PSS |
| ME-59 | `37`, the long SI pointer sentence | five topics listed together | `REWRITE`: cite in two groups, data/model validation and inverse design/DFT/statistics, without repeating the main-text results | points clearly at the reproduction information |

### First-principles verification, line 41

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-60 | `41`, `Four quantities ...: contact bounds ..., ordering ..., damage-severity ..., property target...` | the opening sentence carries both the methodological principle and the list of four | `SPLIT`: the first sentence says four learned quantities were re-derived under a pre-fixed DFT protocol; the second lists the four in parallel, with semicolons if needed | purpose first, then the objects |
| ME-61 | `41`, VASP/PBE/cutoff/k-mesh/EDIFF/smearing/relaxation/five volumes | eight settings in one enormous sentence | `SPLIT`: one sentence for the code, pseudopotential and cutoff; one for the k-points, convergence and smearing; one for the relaxation and volume sampling | lightens the load of technical parameters |
| ME-62 | `41`, `Each first-principles quantity ... follows from those runs.` | metadiscourse adding no new operational information | `CUT/MERGE` into the opening sentence | reduces self-reference |
| ME-63 | `41`, the reduced-contact landscape sentence | clearly defined | `KEEP` | corresponds to the contact bounds |
| ME-64 | `41`, the relaxation-energy release sentence | the definition and the DFT/ML-potential comparability share a sentence | `SPLIT`: one sentence for the definition of the quantity; one for the identical cell-by-cell computation | separates "what it is" from "why it is comparable" |
| ME-65 | `41`, the order-disorder temperature sentence | three nested layers: numerator, denominator and enumeration procedure | `SPLIT`: the energy numerator; the configurational-entropy denominator; the relaxation procedure for symmetry-distinct orderings | makes the estimator reproducible in order |
| ME-66 | `41`, the bulk-modulus sentence | clear | `KEEP` | corresponds to the design property |
| ME-67 | `41`, the 1,917 tasks, SI and 260 CIF sentences | clear, but they are campaign and archive information | `BREAK`: a separate closing paragraph, giving the campaign size, then the SI protocol, then the data availability | separates them from the quantity definitions |

### Autonomous agents, human oversight and reproducibility, line 45

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| ME-68 | `45`, the whole paragraph | seven sentences covering, in turn, the boundaries, the agent actions, the human actions, confirmation, responsibility, records and reproduction; logically complete | `KEEP`; when actually rewriting, check only that `agents/agent instances` is used consistently, and delete no oversight sentence | serves as the transparency statement |

## A6. `si.tex`: the SI wrapper

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| SW-01 | `64-66`, the SI title | must be synchronised word for word with the final main-text title | synchronise word for word once the main title is confirmed; do not write a second version separately | prevents the main-text and SI titles drifting apart |
| SW-02 | `84-86`, `This document provides the data sources ... exact procedures ... full record...` | five kinds of material stacked in succession | `SPLIT`: the first sentence lists only the sources/licences, the definitions and the damage protocol; the second lists the search and testing and the eleven refuted claims | establishes a "definitions and protocol -> audit record" structure |
| SW-03 | `86-89`, `It also documents the data-split incident ... additional tests ... PSS ... applications...` | five layers -- the incident, the reproduction, the validation, PSS and the applications -- in one sentence | `SPLIT`: one sentence for the incident and reproduction; one for the additional physical tests; one for PSS and the applications | gives a map of the SI contents in three sentences |
| SW-04 | `68-82, 91-104`, the authors, affiliations, contents and bibliography wrapper | no linguistic problem | `KEEP` | does not touch the layout control code |

# B. The Supplementary Information, sentence by sentence

What follows reviews only the paper text of `si_body.tex`. Purely numerical table cells, LaTeX control commands and citation keys are not objects of linguistic revision; table titles, captions and explanatory sentences are still listed item by item.

## B1. Supplementary Note S1: the agent record, the research boundaries and the audit

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S1-01 | `42-44`, `This is the sense in which ...` | metadiscourse explains "the sense meant here" first, so the main conclusion arrives too late | `REWRITE` into a direct sentence: autonomy is established by preserved proposals, tests and refutations rather than by post-hoc narration | states the checkable basis directly |
| S1-02 | `48-51`, the investigation dates and `All subsequent investigations...` | the dates and the period repeat in adjacent sentences | `MERGE`: keep the time window once, then cite Table S1 directly | compresses the description of the period |
| S1-03 | `72`, the structure counts and PU subsets in the same cell of Table S1 | two kinds of data with different uses share one cell | `SPLIT` into a structure-law study item and a PU/PSS study item | separates the two analysis lines |
| S1-04 | `75`, the Set 4 and PSS success criteria in one table sentence | the fixed-law and continuous-score criteria are not parallel | `SPLIT` into two rows, each with its own outcome and constraint stated | makes the pre-set criteria easy to check |
| S1-05 | `84-86`, `An attempt strip ... The audit separated...` | the definition of the attempt strip and the lane classification run together | `SPLIT`: define one attempt first; then list the discovery, confirmation and exploration lanes | the unit first, then the classification |
| S1-06 | `86-89`, the 175 attempts, the 131/37/7 and the no-mark explanation | the total, the marked counts and the reason for no mark are crowded into one group of sentences | `SPLIT`: one sentence for the total; one for the three counts; one for the reason there is no mark | states the audit denominator clearly |
| S1-07 | `89-92`, the adjacent-screening exclusion and the optimal-tree marking | two unrelated audit rules | `SPLIT` into two sentences, each stating which class of attempt it affects | keeps the rules from being chained together |
| S1-08 | `94-98`, `What is counted ... scope ... traceability` | definition, scope and traceability in succession | `SPLIT`: the count unit; what is not counted; how it links to the files and logs | defines the attempt count checkably |
| S1-09 | `98-100`, the 2,037,606 and the 8,466 adjacent screens | the total search count and the adjacent screening count are easily taken for the same statistic | `REWRITE` as an explicit contrast: total hypotheses against local follow-up screens | distinguishes the global from the local search |
| S1-10 | `100-104`, the pre-confirmation and post-confirmation failures | the time stage and the nature of the failure are nested | `SPLIT` into two sentences, pre-confirmation failures and post-confirmation checks | establishes a timeline |
| S1-11 | `108-112`, repeated `caught`/`later caught` | the same audit function expressed over and over | keep the first instance; compress the later 11 failures into one result sentence | reduces rhetorical repetition |
| S1-12 | `112-114`, the missing diagnostic -> the checklist | the checklist comes before the missing diagnostic, inverting the causation | `MOVE` the missing diagnostic first; introduce the checklist with `This omission motivated...` | lets the gap lead naturally to the improvement |

## B2. Supplementary Note S2: the data inventory and the split flow

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S2-01 | `175-178`, an arrow chain describing the data flow | the arrow notation mixes the processing flow with the sample coverage | `REWRITE` into two sentences: the first gives the flow in order; the second the coverage at each stage | expresses the data pipeline readably |
| S2-02 | `185-188`, the trajectory frames, the sample counts and the endpoints | trajectory sampling and the endpoint definition share a sentence | `SPLIT`: the frame sampling; the number selected; which endpoint is retained | describes the time-series sampling |
| S2-03 | `192-193`, `Composition of failures:` | a sentence fragment | `REWRITE` into a complete sentence, stating what denominator the following proportions use | fixes the grammar and the statistical reference |
| S2-04 | `214-217`, the coverage definition and the numbers | the definition and the results share a sentence | `SPLIT`: define evaluable coverage first; then report the per-law and per-set figures | the metric before the result |
| S2-05 | `217-220`, the fixed laws applied without refitting, and the comparative outcomes | the protocol and the result share a sentence | `SPLIT`: the first sentence stresses no refitting; the following sentences report the source-specific rates | separates the transfer rule from the performance |

## B3. Supplementary Note S3: descriptor conventions and the eight laws

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S3-01 | `224-226`, the opening fragment, CrystalNN and the valence source | the definition opens with a sentence fragment and two sources are listed together | `REWRITE` into a complete lead-in sentence; a sentence each for the neighbour source and the valence source | establishes a map of the conventions |
| S3-02 | `228-231`, the four Pauling criteria | the four are not syntactically parallel | `REWRITE` into a strictly parallel numbered list, each item "input/condition -> test" | easy for a reader to scan |
| S3-03 | `231-234`, the missing convention placed after the satisfaction rule | the reader sees the verdict before learning how missing values are handled | `MOVE` the missing-input convention before the satisfaction definition | define evaluability before defining a pass |
| S3-04 | around `235-237`, the T0/T1/T3 table appears with no lead-in | the labels get no one-sentence explanation at first appearance | add a sentence before the table explaining the role of T0, T1 and T3; do not change any value | makes the table readable on its own |
| S3-05 | `271-272`, `existing structures` repeats | the same source or set is restated in adjacent sentences | `MERGE` into one precise naming | removes the verbal repetition |
| S3-06 | `279-282`, the missing-value retention reason | the rule, the reason for retention and the consequence for the denominator share a sentence | `SPLIT` into the missing policy, why it is retained, and how it is counted | prevents it being read as imputation |
| S3-07 | `287-289`, `The implementation follows ...` | metadiscourse comes first and the concrete lookup sequence after | `CUT` the empty opening and list the fixed lookup order directly | improves operational clarity |
| S3-08 | `292-296`, the CN fallback, the element-radius fallback and the fixed radius | three nested layers of fallback | `SPLIT` into three sentences or a numbered list in priority order | expresses the hierarchy reproducibly |
| S3-09 | `299-303`, the bond-valence source/key and the parameter selection | the data source and the selection algorithm share a sentence | `SPLIT`: one sentence for the source and key; one for the lookup and selection rule | separates data from algorithm |
| S3-10 | `305-307`, the emitted condition and the average calculation | the output condition and the computation of the statistic share a sentence | `SPLIT` | makes clear when a value exists and how it is computed |
| S3-11 | `308`, metadiscourse such as `implemented verbatim` | adds no definition | `CUT` or merge into the concrete formula sentence | reduces self-description |
| S3-12 | `312-315`, the primary and robustness neighbour algorithms | two sets of algorithms listed in succession | `REWRITE` into one sentence for the primary convention and one for the robustness alternatives | separates the main analysis from the sensitivity analysis clearly |
| S3-13 | `315-317`, the `32x` and three exact values | the factor and the values are crowded together, and `percentage points` is elided | `SPLIT`: report the exact rates first; then explain with the full `percentage points`/fold change | avoids ambiguity of units |
| S3-14 | `317-320`, `does not license...` | stating the conclusion as a withheld licence is convoluted | `REWRITE` into a direct statement of the specific interpretation this robustness check supports | states positively what the evidence means |
| S3-15 | `324-328`, the full definitions and the SI navigation | clear | `KEEP` | closes the conventions |

## B4. Supplementary Note S4: the search defect and its repair

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S4-01 | `338-341`, the model form, the objective and the constraints | three parts of the search definition in one sentence | `SPLIT`: the candidate model; the objective; the constraints | lets the search specification be read in order |
| S4-02 | `345-348`, the search defect and its consequence | the bug mechanism and the statistical consequence are mixed | `SPLIT`: what failed; which candidates remained or were lost | the defect first, then its effect |
| S4-03 | `348-349` and `369-371`, the complete beam allocation repeats | the same resource allocation is described twice | `MOVE` the full description to the defect paragraph; delete the repetition later and keep only the result | removes the duplication |
| S4-04 | `353`, the subheading `--ban did not ban` | colloquial, and its object is only clear after reading the text | `REWRITE` into a direct technical heading: ban-filter failure and correction | the problem is visible at a glance |
| S4-05 | `353-355`, the composite key and the filtering failure | an implementation detail and the outcome share a sentence | `SPLIT` | explains how the bug arose |
| S4-06 | `361-364`, the candidate triple, the per-column handling, the conditional logic and T0/T1 | four layers of logic too densely packed | `SPLIT` into an ordered procedure with T0/T1 marked explicitly | makes the repair steps reproducible |
| S4-07 | `366-369`, the three operating points | the three definitions are not syntactically parallel | `REWRITE` onto one template: threshold source + intended use | makes the operating points comparable |
| S4-08 | `391-395`, the axes, labels and references of Fig. S3b | the caption explains the axes, the colours and the reference lines at once | `SPLIT` into three sentences: panel content, encoding, reference | lets the caption be parsed quickly |

## B5. Supplementary Note S5: the damage controls

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S5-01 | `438-441`, the missing repulsion and the `bl_min` supplement | the missing descriptor and the supplementary contact measure are mixed into one sentence | `SPLIT`: why the primary metric is unavailable; how `bl_min` supplements it | states the basis for the fallback |
| S5-02 | `443-446`, the S6 decision, the two probes and the design meaning | three layers: the decision, the verification and the implication | `SPLIT` into decision -> probes -> implication | shows how the control supports the design |
| S5-03 | `446-447`, the R1/S6 distinction | repeats the difference stated in the previous sentence | `MERGE` into the last sentence of S5-02 | explains the two controls once |
| S5-04 | `449-451`, the topic, geometric and electrostatic labels | the three explanatory terms have no parallel grammar | `REWRITE` into three parallel items | names the physical object of each probe |
| S5-05 | `451-454`, the D5 gain and the maximum for the other classes | the main result and the comparison baseline share a sentence | `SPLIT`: the D5 improvement; the largest non-D5 change | foregrounds the specificity |

## B6. Supplementary Note S6: the refuted claims

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S6-01 | `485-492`, the complete ledger, Fig. S1, Fig. S2 and the machine ledger | "complete record" repeats while the functions of the three carriers are not distinguished | `MERGE/REWRITE`: Fig. S1 = overview, Fig. S2 = timeline, the machine ledger = full trace | describes the three layers of evidence in one statement |
| S6-02 | `496-498`, the equality observation and the past interpretation | the observation and the interpretation later rejected share a sentence | `SPLIT` | separates the fact from the refuted inference |
| S6-03 | `509-513`, the general principle, the Pauling example and the decision/tie rates | three layers: principle, example, numbers | `SPLIT` into three sentences | moves from the refutation rule to the illustration |
| S6-04 | `513-517`, the scoring error, the corrected result and the symmetry refutation | two different refutations run together | `SPLIT/BREAK`: the correction and the result first; then the refutation of the symmetry claim | keeps the cases from being conflated |

## B7. Supplementary Note S7: the distance-cutoff benchmark

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S7-01 | `524-525`, the benchmark setup fragment | a sentence fragment | `REWRITE` into a complete subject-predicate sentence naming the sample and the methods compared | makes the benchmark understandable on its own |
| S7-02 | the remaining paragraphs of S7 | the sample, the cutoffs and the Set 1/Set 4 matched comparison are clearly structured | `KEEP` | no unnecessary rewriting |

## B8. Supplementary Note S8: the protocol, the split incident and the reserve

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S8-01 | `560-563`, the preregistration/tag/commit/seed fragments | noun phrases rather than complete sentences | `REWRITE` into two or three complete statements: the protocol location and version; the code commit; the seed and split assignment | gives the reproducibility items a predicate |
| S8-02 | `568-569`, R1/R2 and the timing of the change | the roles of the two revisions share a sentence | `SPLIT`: the content of R1; the content and timing of R2 | establishes a protocol timeline |
| S8-03 | `579-581`, the authorization date and the V4 definition | the governance event and the model-version definition are mixed | `SPLIT` | separates governance from the object of analysis |
| S8-04 | `581-583`, the two success criteria | the two are not syntactically parallel | `REWRITE` onto the same template with the numerator and threshold stated explicitly | makes them auditable |
| S8-05 | `583-587`, the three effects, the direction and the decision | the three consequences of the incident are packed into one group of sentences | `SPLIT`: effect 1/2/3; the direction; the decision | makes the chain of consequences explicit |
| S8-06 | `588-591`, the control values and the mechanism | the numbers and why they constitute a control share a sentence | `SPLIT`: the numbers first; then the mechanistic explanation | the reader sees the evidence before its meaning |
| S8-07 | `592-596`, the three datasets, the meaning of the method and the remaining reserve | the datasets and the independence conclusion are too dense | `SPLIT`: name the discovery, held-out and reserve status separately; a final sentence saying what remains valid | states the current standing of each split clearly |

## B9. Supplementary Note S9: reproduction

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S9-01 | `611-638`, the commands and the public benchmark | the command comments are concise and the benchmark licence sentence is clear | `KEEP` | does not change an executable command or the meaning of a licence |

## B10. Supplementary Note S10: the exact shallow-tree search

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S10-01 | `642-643`, Purpose | clear | `KEEP` | defines the exact-search question |
| S10-02 | `645-648`, the discovery matrix, STreeD and the quantile binarization | the data size, the full algorithm name and the parameters are too dense in one sentence | `SPLIT`: one sentence for the matrix; one for the algorithm and class; one for the binarization settings | data first, then algorithm, then discretisation |
| S10-03 | `648-651`, the candidate splits, the search class and the runtime | the principle, the model-class relation and the runtime are crowded into one paragraph | `BREAK`: a sentence each for the meaning of the quantiles, the class containment and the runtime | states the scope of the exactness and its computational cost |
| S10-04 | `653-656`, the objectives, the identical tree, the cost ratio and the sample replication | objective invariance and the cost-ratio control share a group of sentences | `SPLIT`: the F1/accuracy result; what moves the curve; how it is implemented | explains the trade-off control clearly |
| S10-05 | `677-678`, the omitted-class test setup | clear | `KEEP` | leads into the table |
| S10-06 | `697-702`, the single threshold, the mean absolute change, the signed-mean cancellation and the interpretation | four layers of comparison in succession | `SPLIT`: the threshold rates; the definition of mean absolute change; the tree and threshold values; why the signed mean misleads | explains the stability metric step by step |
| S10-07 | `704-708`, the tree features, the loss specificity and the interpretation after `Two corroborating details` | the two details are not numbered and the first list is very long | `REWRITE` into two `First`/`Second` paragraphs; keep the feature list in one place; the second paragraph explains the class-specific loss | makes the heading match the text |
| S10-08 | `710-717`, the replication and the supported conclusion | both paragraphs are clear | `KEEP` | reports the independent replication and the strong within-model-class conclusion |

## B11. Supplementary Note S11: the one-sided contact criterion

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S11-01 | `721-725`, what a lower bound misses | clear | `KEEP` | defines the expansion blind spot |
| S11-02 | `727-729`, the grid scan and matched-satisfaction fairness | the scan range and the fairness principle share a sentence | `SPLIT`: the grid definition; the comparison at the same satisfaction | describes the search first, then the comparison |
| S11-03 | `750-752`, the upper bound snapping, and the 0.99-row error/R11 | the threshold behaviour and the historical error share a group of sentences | `SPLIT`: the first sentence reports only the abrupt change; the second explains that reporting only the 0.99 row led to R11 | moves from result to refutation |
| S11-04 | `754-755`, the heading `What the upper tail actually is` and its sentence | the heading is colloquial and the sentence is convoluted with "not exotic ... but ..." | `REWRITE` the heading as `Chemical character of the upper tail`; state directly that the upper-tail structures lie outside the ionic domain | names the finding directly |
| S11-05 | `774-776`, radius-model applicability, the charge filter and the anion figure | the three sentences jump a little | `MERGE/REWRITE`: say that charge balance alone does not establish applicability of the ionic domain; then cite the anion analysis | connects the domain mismatch to the analysis that follows |
| S11-06 | `778-783`, `Hence` runs straight into the displayed law | `Hence` forms a sentence fragment | `REWRITE` into a complete lead-in: these distributions motivate the conditional law below | introduces the formula with complete grammar |
| S11-07 | `823-824`, the full law-set sequence | clear | `KEEP` | closes the expansion result |
| S11-08 | `826-827`, `Selecting ... only:` | a sentence fragment before the colon | `REWRITE` into a complete sentence stating that the condition and threshold were selected on the four retained classes | describes the LOFO setup on its own |
| S11-09 | `860-862`, the same threshold, the stability and the omitted result | threshold stability and the meaning of the performance share a sentence | `SPLIT`: the same threshold across five repeats; the omitted-type result and what it isolates | separates selection stability from transfer |
| S11-10 | `864-868`, the stated limitation | the four sentences are logically clear | `KEEP`, making `It` refer explicitly to the leave-one-type test | avoids the ambiguous pronoun |
| S11-11 | Figs. S7-S10, `873-916` | the captions are clear, but some single sentences carry the axes, the encodings and the reference lines | lightly `SPLIT` each panel onto a three-sentence template, "what is plotted -> encoding -> reference"; change no figure meaning | gives the SI captions a uniform rhythm |
| S11-12 | Fig. S11 panel a, `923-931` | the compounds, the axis, the colour, three reference bounds and the conclusion share a paragraph | `SPLIT` into four sentences: data and axes; colour; the horizontal and vertical references; the observed rise | the reader reads the figure before the conclusion |
| S11-13 | Fig. S11 panel b, `931-935` | the potentials, both axes, the diagonal and the annotation in one sentence | `SPLIT` into the comparison; the axes and diagonal; the annotation | explains the hard-potential control |
| S11-14 | Fig. S11 panel c, `935-940` | the crossing definition, the normalisation, the boxes and the 1.80x interpretation run on without a break | `SPLIT` into the coordinates; the normalisation; the box encoding; the 1.80x result and its meaning | foregrounds the core result in the caption |
| S11-15 | Figs. S12-S13, `947-969` | essentially clear | `KEEP`; only harmonise how the panel sentences begin | no substantive rewriting |

## B12. Supplementary Note S12: the damage operators

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S12-01 | `974-995`, the D1-D5 procedure table | the table is clearly structured | `KEEP` | preserves the exact operators |
| S12-02 | `998-1001`, the wrong D2 swap, the unchanged charges, the zero energy and the relabelling | the implementation error, the null consequence and the interpretation share an over-long sentence | `SPLIT`: the original error; the zero-energy result; why it was only relabelling | reconstructs the R1 causal chain clearly |

## B13. Supplementary Note S13: composition-grouped ranking

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S13-01 | `1005-1007`, the pairing rule, the group/structure/pair counts and the eligibility | the rule, the sample sizes and the restriction share a sentence | `SPLIT`: the pairing rule; the ionic restriction; the counts | defines the ranking cohort |
| S13-02 | `1009-1013`, the usable comparison, the tie, the pair decision rate and the group tie rate | pair-level and group-level definitions run in succession | `SPLIT/BREAK`: the pair verdict definitions; the pair decision rate; the group tie rate | prevents the two levels being confused |
| S13-03 | `1012-1017`, accuracy, ties, Top-1 hit, baseline and lift | four metrics densely packed in one paragraph | `SPLIT`: conditional accuracy; tie handling; the Top-1 definition and baseline; lift | one sentence per metric |
| S13-04 | `1019-1022`, the within-group statistics, the equal weighting and the different contributing groups | the aggregation and the non-common cohort share a sentence | `SPLIT`: the group-wise calculation; the equal weighting; the criterion-specific contributing groups | states the difference in denominator |
| S13-05 | `1022`, `Both conventions are essential...` | `Both` is unclear, since more than two conventions precede it | `REWRITE` to name tie handling and group-equal weighting directly | removes the ambiguous pronoun |
| S13-06 | `1022-1026`, the 77.7% tie error and the SiO2 pair-weighting reversal | two different error cases crowded into one paragraph | `BREAK`: one paragraph for the R10 tie-scoring; one for the R9 SiO2 weighting; each giving the error before the corrected comparison | keeps each audit case separate |
| S13-07 | `1028-1031`, the cluster bootstrap, the reason and the consequence of pair resampling | the method and the reason for choosing it share an over-long sentence | `SPLIT`: the resampling unit and procedure; the within-group correlation reason; the consequence of pair resampling | makes clear why a cluster bootstrap |
| S13-08 | `1031-1032`, `Folds are split... B=500, seed...` | the last sentence is a fragment of parameters | `REWRITE` into a complete sentence, separating the fold split from the bootstrap settings | gives the reproducibility information complete grammar |

## B14. Supplementary Note S14: figure conventions

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S14-01 | `1034-1037`, the whole paragraph | three concise sentences | `KEEP` | describes the figure style and the source data |

## B15. Supplementary Note S15: PSS fitting, evaluation and stability

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S15-01 | `1041-1047`, the definition of PSS and why a new analysis | clear | `KEEP` | moves from the binary law to continuous ranking |
| S15-02 | `1049-1052`, the programme relation, the date, the reserve event, the reuse and the statistics | the lineage and reuse information are too dense | `SPLIT`: one sentence for the same programme and date; one for the relation to the reserve event; one for the reused data and code; one for the inclusion of the statistics | states the provenance clearly |
| S15-03 | `1054-1058`, the initial plan, revision 1 and revision 2 | the two revisions are packed into one long sentence | `SPLIT`: a sentence each for the initial preregistration, the R1 trigger and change, and the R2 change and timing | establishes a version timeline |
| S15-04 | `1058-1060`, the reproduction of the four baselines | clear | `KEEP` | states the reproduction preceding any new row |
| S15-05 | `1062-1064`, the development and held-out groups, structures and pairs | three counts for two splits strung together with slashes | `SPLIT` into a development sentence and a held-out sentence, or a small table | maps the split counts at a glance |
| S15-06 | `1064-1068`, the SiO2 placement, the dominance implication and the single evaluation | three clear layers, but they should be separate paragraphs | `BREAK`: the composition-dominance control; a standalone sentence for the one-shot held-out evaluation | separates a property of the cohort from evaluation discipline |
| S15-07 | `1070-1077`, the 85-feature exclusions and the nine additions | the exclusions, the additions and the list of nine features share a sentence | `SPLIT`: one sentence for the excluded families; one for the added symmetry features; one for the repulsion/strain/density features | organises the feature pool by physical function |
| S15-08 | `1077-1079`, no DFT, and the charge and radius conventions | clear | `KEEP` | defines the low-cost inputs |
| S15-09 | `1081-1086`, the model class, standardisation, weighting, forward selection and refit/archive | the procedure is right but five steps are dense in one paragraph | `SPLIT` into the model form; the preprocessing and weighting; the selection and stopping; the final refit and hash | presents the training pipeline in order |
| S15-10 | `1088-1092`, the forward-selection sequence and the stopping rule | six numbered steps in one long colon-led string | `REWRITE` into a numbered or semicolon list, with a final sentence saying the seventh candidate falls below 0.002 | presents the increments scannably |
| S15-11 | `1092-1097`, the all-feature model, the hull, the archive and the leading singles | the benchmark results and the archive inventory are mixed | `SPLIT/BREAK`: the model and hull scores; the archive statement; the leading single-feature audit | separates the performance comparison from the reproduction material |
| S15-12 | `1099-1100`, the experimental records | clear | `KEEP` | refers back to the S13 label convention |
| S15-13 | `1102-1108`, the PSS/full/hybrid scores, the reference scores and the bootstrap differences | six sets of results stacked in one paragraph | `SPLIT`: the three model scores; the four baselines; the paired differences | models first, then baselines, then uncertainty |
| S15-14 | `1108-1112`, the decision rate, the best structural single, Top-1, not overtaking the hull, and the wording | five conclusions in one group of sentences | `SPLIT`: the decision rate and structural comparison; the Top-1 comparison; the overall hull comparison; the main-text wording | keeps every strong result without swallowing a number |
| S15-15 | `1112-1114`, the development-to-held-out drop and the evaluation log | the explanation and the audit record are clear | `KEEP` | describes repeated-development optimism |
| S15-16 | `1116-1119`, the confidence heading, and the metric/CI/archive in parentheses | key statistical definitions are buried in parentheses | `REWRITE`: shorten the heading; state the ordering metric, the pair accuracy, the cluster CI and the archive separately in the text | brings the definitions of the analysis into the main clauses |
| S15-17 | `1137-1140`, the post-hoc status, CeSe2, and the all-pair/group-equal difference | the temporal status, the composition share and the aggregation explanation run in succession | `SPLIT`: the descriptive status; the largest composition; why all-pair differs | marks the exploratory breakdown clearly |
| S15-18 | `1141-1145`, the one interval, Bonferroni and removing the composition | two robustness objections in one long sentence | `SPLIT`: the primary fifth result; the multiplicity check; the composition-removal check | gives each robustness check its own sentence |
| S15-19 | `1145-1147`, the shares of the five largest compositions | five bare proportions not mapped to pair fractions | `REWRITE` into an explicit `100%, 50%, 30%, 20%, 10% -> shares` mapping | the reader no longer has to guess the order |
| S15-20 | `1147`, the exploratory status | clear | `KEEP` | closes this breakdown |
| S15-21 | `1149-1153`, the fixed terms, the coefficient-only refit, the development counts and the held-out not being reused | two scope boundaries in one paragraph, which could be clearer | `SPLIT`: the first sentence says the feature search was not reopened; the second that only development groups were used; the third that the held-out was not re-evaluated | locks down the scope of the analysis clearly |
| S15-22 | `1155-1159`, the coefficient movement, the rank stability and the cosine stability | the summary and two stability statistics share a sentence | `SPLIT`: the coefficient movement; the score-rank correlation; the vector-direction similarity | separates prediction stability from parameter stability |
| S15-23 | `1159-1163`, the looseness of the magnitudes, two examples and the retention of sign | three numeric layers in one sentence | `SPLIT`: the variability of the magnitudes; the eta/Mz contrast; the frequency of the sign | foregrounds which quantities are stable |
| S15-24 | `1163-1168`, the cluster bootstrap, the term ordering, the ratio range and the sign | the method and three results are too dense | `SPLIT`: the bootstrap method; the signal-to-error ordering; the consistency of sign | reports the second stability analysis clearly |
| S15-25 | `1168-1170`, what the data fix, and the decimal warning | a strong summary, and clear | `KEEP`; change `parts` to the concrete `signs and relative weights` as the subject | gives the interpretation directly |
| S15-26 | Fig. S14 panel a, `1175-1180` | the fixed-term setup and the box/median/whisker/dashed line run in succession | `SPLIT`: the setup; the distribution encoding; the reference line | lightens the caption |
| S15-27 | Fig. S14 panel b, `1181-1183` | the bootstrap interval, the published values and the right-hand labels in one sentence | `SPLIT` into the interval and procedure; the markers; the labels | separates the three visual encodings |
| S15-28 | Fig. S14 panel c, `1184-1187` | two agreement metrics and the lines and shading in one sentence | `SPLIT`: the rank and vector metrics; the mean line; the interval shading | separates metric from encoding |
| S15-29 | `1191-1196`, the structure-level correlations and the within-composition diagnostics | two scales of analysis crowded into one paragraph | `SPLIT/BREAK`: the across-structure correlations; the within-composition correlations, VIF and condition number | the overall picture first, then the actual fit space |
| S15-30 | `1197-1199`, the spread inference | clear, but `spread` needs a referent | `REWRITE` as `The coefficient spread...` | removes the burden of referring back to a figure number |
| S15-31 | Fig. S15, `1204-1211` | panels a and b are clear | `KEEP`; only harmonise the parallel grammar of `Entries above/below` | no substantive rewriting |

## B16. PU transfer, the property screen and inverse design, lines 1215-1393

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| PU-01 | `1217-1229`, the Jang lineage, the later application, the expanded positives and unlabelled set, the two models, the consensus and the deduplication | one paragraph crosses the literature history, this work's data, the models and the screening result; the storyline is too dense | `BREAK` into three paragraphs: the prior lineage; the current pool expansion; the two-model consensus and deduplication | background first, then this study, then the hard-negative cohort |
| PU-02 | `1217-1222`, the Jang et al. sentence and the later application | clear | `KEEP`, saying `the pretrained CLscore model` explicitly in the second sentence | maintains the timeline |
| PU-03 | `1222-1225`, the expansion of the positive and unlabelled sets | two pools and four sources in succession | `SPLIT`: the positive set and the ICSD/COD counts; the unlabelled set and the LeMat/ELEMENTA count | separates labelled from unlabelled |
| PU-04 | `1225-1229`, the 50-bag CGCNN, the 50 MLP heads, the consensus 364,771 and the deduplicated 364,592 | model construction and cohort construction share a paragraph | `SPLIT`: the two models; the consensus rule and result; the deduplication and result | builds model -> selection -> final cohort |
| PU-05 | `1231-1234`, the four Set rates presented as eight numbers in succession | hard for a reader to align satisfaction and screening with Set 1-4 | `REWRITE` into a small table or four strictly parallel clauses, each in a fixed `satisfaction; screening` order | shows the law-set progression clearly |
| PU-06 | `1234-1235`, the PSS curve and the matched point | clear | `KEEP` | gives the PSS matched-satisfaction result |
| PU-07 | `1235-1238`, descriptor completeness and median imputation | the coverage figures and the missing-value rule are mixed into the results paragraph | `BREAK`: one sentence for the observed coverage; one for the frozen-median rule | separates the result from the evaluation convention |
| PU-08 | `1240-1244`, fractional attribution and leave-one-law-out | two attribution methods in succession, without their complementarity being stated first | `SPLIT`: the definition and normalisation of the fractional allocation; the LOO definition | explains contribution and necessity separately |
| PU-09 | `1244-1248`, the Law 7 share, the removal impact and the Law 1 task contrast | the numbers and the cross-task explanation run in succession | `SPLIT`: the fractional share; the LOO loss and recovery; the controlled-damage contrast | moves from the result to the task dependence |
| PU-10 | Fig. S16, `1253-1260` | the panel a/b definitions are clear | `KEEP`; move `Records without a verdict...` to the common-cohort note at the start of the caption | states the shared denominator first |
| PU-11 | `1264-1267`, the two PU representations | both model definitions are clear | `KEEP both sentences` | explains the complementary encodings |
| PU-12 | `1267-1274`, the validation bags, the AUCs, the bag sizes and the figure contents | the evaluation protocol, the performance and the sample size are too dense in one paragraph | `SPLIT`: the bag protocol; the two AUCs; the CGCNN bag size; the MatterSim availability; the figure navigation | follows protocol -> result -> cohort |
| PU-13 | Fig. S17, `1279-1287` | panels a/c and b/d are each clear | `KEEP`; move the shared bag-composition sentence before panels a/c | the common setup precedes the panel details |
| PU-14 | Fig. S18, `1294-1299`, the axes, the 260/261 and the non-convergence | the figure definition and the missing candidate share a paragraph | `SPLIT`: the axes and fits; the cohort shown; the one non-converged cell | establishes the denominator |
| PU-15 | Fig. S18, `1299-1301`, below the diagonal, the 0.940 factor and the 400 -> 376 thresholds | the `because` clause strings together the direction, the scale factor and both thresholds | `SPLIT`: the systematic scale relation; the median factor; the calibration mapping | explains the two scales clearly |
| PU-16 | Fig. S18, `1301-1304`, the 0.966 ranking, the 3/140 removed and the Re2IrOs6 exception | the overall result and the exception share a paragraph | `SPLIT`: the ranking retention; the top-140 count; the named exception | foregrounds both the overall result and the counterexample |
| PU-17 | Fig. S18, `1305-1306`, the transfer note | clear | `KEEP` | qualifies the cohort sampling without weakening the main result |
| PU-18 | Fig. S19, `1313-1321` | all three panel definitions are clear | `KEEP`; put the tie convention inside the panel c sentence | lets each panel be understood on its own |
| PU-19 | Fig. S20, `1328-1330`, `All 260 candidates...` after the caption title | the second sentence is a fragment with no predicate | `MERGE` into the caption opening: the panels show all 260 candidates | fixes the fragment |
| PU-20 | Fig. S20 panel a, `1330-1332` | the plot and the composition conclusion are clear | `KEEP` | shows the property conditioning |
| PU-21 | Fig. S20 panels b/c, `1333-1339` | the crystal system, the Law 7 quantity, the colour, the diagonal, the bound and the data availability run in succession | `SPLIT`: the panel b definition and convention; the panel c axes and colour; the meaning of the diagonal; the bounds; the archive | separates the before-after symmetry from the visual encoding |
| PU-22 | `1344-1349`, the CLscore relation | the procedure, the two endpoint trends and the cross-model inference are clear in four sentences | `KEEP`; complete the first sentence with why the raw scales are not averaged | explains the rationale for percentile alignment |
| PU-23 | `1351-1354`, the balanced pilot, the finite energies, the hull reference and the sweep | four processing steps in succession | `SPLIT` into the sample; the relaxation success; the hull construction; the threshold sweep | makes the energy route reproducible |
| PU-24 | `1355-1357`, the 0.20-eV result and the cost inference | clear | `KEEP`, making the subjects of retained and screened parallel | compares the energy screen with PRIS |
| PU-25 | `1359-1364`, the MatterGen version and commit, the 13 shards, the parsed count and the deduplication parameters and result | the generation and deduplication information is too dense | `BREAK`: the generation settings and count; the StructureMatcher settings; the deduplication outcome | establishes the candidate cohort |
| PU-26 | `1365-1369`, the UMA proxy, the five volumes, the quadratic fit, all passing and the 140 high-property candidates | five steps of the property calculation in succession | `SPLIT`: the checkpoint and task; the volume sampling; the fit criterion; the success and high-property counts | explains how the proxy is derived |
| PU-27 | `1369-1374`, the frozen PSS, the 2 of 6 descriptors, the 541-candidate calibration cohort, the 97.5% target and the cutoff | the state of the score, the support matching and the cutoff calibration are mixed | `SPLIT/BREAK`: the frozen score and imputation; the calibration cohort; the target satisfaction; the selected cutoff | makes clear that the cutoff used no generated labels |
| PU-28 | `1374-1378`, the screening/tie rule, the 97.6%/61/140, no generated data and the support stratum | the decision rule, the result and the leakage control run in succession | `SPLIT`: the screening rule; the calibration outcome; the 61/140 application result; no generated data in the fitting; the support-only role | shows calibration -> application -> control |
| PU-29 | `1378-1382`, the two-descriptor equation | clear | `KEEP` | gives the exact form for the matched stratum |
| PU-30 | `1383-1386`, the site fraction, distance pass, Law 7, volume range and means, and the subset relation of the 61 candidates | five characteristics stacked together | `SPLIT`: the shared verdict profile; the screened volume distribution; the retained mean comparison; the subset relation | describes the PSS-removed cohort |
| PU-31 | `1387-1388`, the 95/140 removed by Set 4 and the denser packing retained by PSS | the strong contrast is clear | `KEEP` | states the fixed-law/PSS division of labour |
| PU-32 | `1388-1392`, the matched Ir2Os7 examples and the descriptors and properties of the two structures | the two examples use the same template but are crowded at the end of the paragraph | `BREAK`: one sentence for the screened structure and one for the retained one, in the same field order | forms a direct matched example |
| PU-33 | `1392-1393`, the archive inventory | five archive items in succession | `SPLIT` into two groups, the generation/model artefacts and the evaluation artefacts | describes the reproduction material clearly |

## B17. Supplementary Note S17: Laws 7/8, external structures and mechanistic diagnosis

### The S17 opening, the selection and the held-out evaluation, lines 1395-1450

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S17-01 | `1395`, the section title | the title crowds two findings and the sequence together, and is too long | `REWRITE` into the shorter `Extending the plausibility sequence with structural simplicity and bond valence`, without changing the conclusion | makes the subject of the section immediately recognisable |
| S17-02 | `1397-1402`, the preregistration, the hash, the data reuse, the recreation and the Set 3 reproduction | the protocol lock, the data provenance and five gate steps share a paragraph | `SPLIT`: the plan and hash; the reserve-free tables and recreation; the Set 3 reproduction gate | establishes the preconditions of the search |
| S17-03 | `1403`, `Model class: Set 3 unchanged...` | a sentence fragment | `REWRITE` into a complete sentence beginning `The model class kept...` | fixes the grammar |
| S17-04 | `1403-1406`, the model class and the 8,466 candidates over 94 quantities | the rule form, the threshold grid and the candidate count are too dense | `SPLIT`: the permitted additions; the threshold source; the candidate count and eligible quantities | defines the fixed search space |
| S17-05 | `1406-1409`, the objective, the one-shot held-out evaluation and the previous reuse | the selection objective and the evaluation governance share a paragraph | `BREAK`: the objective and constraint; the held-out evaluation; the S8 disclosure | separates fitting from evaluation history |
| S17-06 | `1411-1416`, the Law 7 bullet | the definition, the Pauling meaning, the grid percentile and the population percentile in one long bullet | `SPLIT` into the threshold law; the physical interpretation; the provenance of the threshold | three layers of information for one law |
| S17-07 | `1417-1419`, the Law 8 bullet | the definition and the principle share a sentence | `SPLIT` into the hard bound and the physical meaning | parallels the structure used for Law 7 |
| S17-08 | `1421`, `Discovery: satisfaction..., damage detection...` | a sentence fragment | `REWRITE` into a complete result sentence | reports the discovery performance |
| S17-09 | `1421-1422`, the gain of the third candidate and the stopping rule | clear but the sentence is long | `SPLIT`: the value of the gain; below the pre-fixed threshold, therefore stopped | makes the stopping rule explicit |
| S17-10 | `1424-1430`, the overall rates, the five per-type rates, the three experimental outcomes, the three damaged outcomes and the criteria | twelve numbers in one paragraph | `REWRITE` into a small table or five sentences: overall; by damage type; the experimental outcomes; the damaged outcomes; the criteria and the margin met | makes the held-out evaluation checkable |
| S17-11 | `1432-1438`, the five LOFO values, the same laws in three folds and the alternate laws in two folds | the list of results and the selection pattern run in succession | `SPLIT`: the five unseen-type results; the three same-law folds; the two alternate-law folds | separates the performance from the identity of the model |
| S17-12 | `1438-1441`, the physicochemical interpretation and the tree-comparison caveat | the inference and the non-direct comparator share a group of sentences | `BREAK`: keep the physicochemical conclusion first; then state the protocol difference between Set 3-fixed additions and a full tree refit | keeps the reader from comparing 0.2318 as if the conventions matched |
| S17-13 | `1443-1447`, the further population, the 440/2024, the parent provenance and the partition counts and non-disjointness | the benchmark design and the provenance share a paragraph | `SPLIT`: the application without refitting, and the sample; the origin of the parents; the partition overlap | the object of the analysis first, then the data relations |
| S17-14 | `1447-1450`, the overall and by-type results, the held-out consistency, the 27x and the source data | seven numbers and two conclusions in one sentence | `SPLIT`: the overall rates; the by-type rates; the held-out consistency; the 27x comparison; the data availability | foregrounds the strong external benchmark result |

### The Law 7/8 division of labour, matched satisfaction and amplitude, lines 1452-1533

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S17-15 | `1454-1458`, the overall and by-type rates of Set 3+Law 7 and Set 3+Law 8 | one sentence carries two sets of seven numbers each | `REWRITE` into a two-row table or two fully parallel sentences, fixing the order satisfaction -> overall detection -> D1-D5 | compares the two additions directly |
| S17-16 | `1459-1460`, the mechanistic contributions of Law 7 and Law 8 | the two sentences are parallel and clear | `KEEP` | summarises the complementary division of labour |
| S17-17 | `1460-1461`, `Two facts ... must be stated plainly` | placeholder metadiscourse | `CUT` and open the first result with `First` directly | gets to the substance faster |
| S17-18 | `1461-1466`, the Law 7 class sensitivity, R5, the combination and the experimental merge control | the detection pattern, the earlier test, the role in the law set and the control are mixed | `SPLIT/BREAK`: the class-specific response; the relation to R5; why the combined set works; the pointer to the merge control | explains the role of Law 7 fully without repetition |
| S17-19 | `1467-1471`, the symmetry tolerance, the flip rates, the symmetrized input and the noisy structures | the convention, the robustness result and the practical advice make one over-long sentence | `SPLIT`: the primary convention; the flip rates at 0.1 tolerance; the input assumption; the practice with refinement noise | moves from definition to deployment |
| S17-20 | `1472-1475`, the D5-omitted refit selecting a space-group law, and the family-level inference | the result and the explanation share a sentence | `SPLIT`: which law is selected; what that shows about the family against an exact threshold | strengthens the point about search recurrence |
| S17-21 | `1479-1484`, the distribution summary, the grid percentile and the GII/practice convention | three scales in succession | `SPLIT`: the distribution quantiles; the location of the threshold; the GII comparison | says where the threshold sits |
| S17-22 | `1484-1488`, the permissive threshold, the separation of the distributions, the two medians and the exclusion of the extreme | the summary, the evidence and the conclusion advance repetitively | `MERGE/SPLIT`: describe the permissive upper-tail bound qualitatively first; then give the medians and separation; then the excluded extreme tail | completes "permissive yet discriminating" in one pass |
| S17-23 | `1489-1490`, the missing BVS parameters | clear | `KEEP` | reports the coverage rule |
| S17-24 | `1494-1497`, the five required satisfactions, the damage-detection range and the 0.81 interpretation | the grid, the result and the inference share a group of sentences | `SPLIT`: the satisfaction levels tested; the damage range; what 0.81 primarily sets | shows the threshold sensitivity |
| S17-25 | `1497-1500`, the exact pair only at 0.81, the Law 8 variants and the space-group replacements | three selection outcomes in one sentence | `SPLIT`: the exact pair; the recurrence of Law 8; the recurrence of the symmetry family | demonstrates family stability |
| S17-26 | `1500-1505`, excluding all the Pauling families against excluding symmetry only | the excluded features, the scores and the selected mechanisms of both ablations are too dense | `SPLIT` into two paragraphs, the all-family exclusion and the symmetry-only exclusion, keeping the fields parallel | compares the ablations clearly |
| S17-27 | `1505-1507`, the final interpretation | the conclusion is clear but `and` joins two claims | `SPLIT`: the Pauling families are not necessary for high detection; a free search selects them first and they give the final gain | keeps the strong conclusion and improves the rhythm |
| S17-28 | `1511-1515`, the 27x at differing satisfaction, the matched cutoff 0.339 against 0.879 and 2.6x, and the Set 1 2.1x | the caveat, the setup and the results are too dense across three layers | `SPLIT`: why matching is needed; the tuned cutoff and result; the Set 4 margin; the Set 1 margin | fairness first, then the strong contrast |
| S17-29 | `1516-1517`, the two bootstrap intervals | both intervals in one sentence | `SPLIT`: the satisfaction CI; the damage-detection CI | pairs each with its endpoint |
| S17-30 | `1518-1521`, the reserve-parent result and why it is not untouched | the numbers and the split status share a paragraph | `SPLIT`: the reserve-parent rates; the implication of the S8 incident | reports the numbers without confusing independence |
| S17-31 | `1525-1529`, the single amplitude, the graded repeats, the onset and the distance-filter result | the setup and both responses make one over-long sentence | `SPLIT`: why a graded test; which perturbations and population; the PRIS onset; the distance-cutoff response | establishes the amplitude-response comparison |
| S17-32 | `1530-1533`, the saturation of the Gaussian displacement, the Wyckoff cause and the law-set progression | the response and cause, and the broader sequence, share a paragraph | `SPLIT`: the Law 7 saturation and its cause; the progressive response of Laws 1-3 | separates symmetry sensitivity from the behaviour of the sequence |

### Generator outputs and GNoME, lines 1535-1646

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S17-33 | `1537-1540`, why an external test, the provenance of the MatterGen sample and the distance filters | the motivation, the sample and the baseline share a paragraph | `SPLIT`: the new external question; the sample and source; the distance-filter result | introduces the generator test |
| S17-34 | `1541-1543`, the full-sample Set 1-4 fractions and the denominator convention | four rates plus the no-verdict rule | `REWRITE` into a one-row progression table, with the next sentence defining the denominator and no verdict separately | prevents a no verdict being read as a fail |
| S17-35 | `1544-1546`, the 338 unassigned and the assignable 4/155/3 | the charge coverage and the conditional outcomes run in succession | `SPLIT`: the unassignable count; the assignable denominator; the three Set 4 outcomes | makes the two levels of denominator explicit |
| S17-36 | `1547-1549`, none failing Law 1 -> the source in the later laws | the observation and the explanation are clear | `KEEP` | points at the chemical, electrostatic and site-order bounds |
| S17-37 | `1549-1553`, no DFT, what satisfaction means, the potential check and the charge limit | four short sentences, but the closing repeats the limitation | `MERGE`: no DFT plus the later energy check; keep the charge assignment as the coverage limit | closes the external screen |
| S17-38 | `1558-1562`, the release and download, the 554,054 and the sample, seed and timing before parsing | the acquisition and the sampling protocol share a paragraph | `SPLIT`: the release, source and size; the uniform sample, seed and timing | the population first, then the sample |
| S17-39 | `1562-1567`, the primitive cell, spglib and deposited label, the difference from discovery, the comparability and the parse success | the symmetry convention and its consequences across four layers | `SPLIT`: the orbit calculation; the label not being used; the difference in cell convention; the consequence for comparing rates; the parse success | marks the external convention clearly |
| S17-40 | `1569-1572`, the overall Law 7 failure and the per-space-group list | the overall figure and six categories are stacked | `SPLIT`: the overall rate; P1/Pm/Cm; the remaining groups | shows where the anomaly concentrates |
| S17-41 | `1572-1577`, the tolerance sensitivity concern, the test and the result | the robustness result comes first and the concern afterwards | `REORDER`: the concern; the strict-to-coarse test; the change in failure rate; the label-agreement result | follows question -> test -> result |
| S17-42 | `1577-1580`, the concentration conclusion, the median and bound, and the archive | the conclusion, the distribution summary and the availability run in succession | `SPLIT`: the space-group anomaly conclusion; the median against the bound; the archive | closes the GNoME prevalence |
| S17-43 | `1582-1589`, the merge hypothesis and the expected positive and negative outcomes | the hypothesis takes three sentences and repeats "similar/distinct" | `COMPRESS/MERGE`: one sentence defining the over-ordering hypothesis; one giving the mergeable prediction; one giving the geometrically distinct control prediction | establishes the logic of the intervention clearly |
| S17-44 | `1589-1593`, the 300-structure stratified sample, the four space groups, the merge class and the recomputation | the sample design and the operator share a sentence | `SPLIT`: the sample strata and count; the merge rule; the recomputed quantities and convention | makes the merge test reproducible |
| S17-45 | `1594-1596`, the mergeable prevalence, 113/150 against 78/150 | clear | `KEEP` | describes the merge opportunity |
| S17-46 | `1596-1603`, little change at strict tolerance, the 78/79% at loose tolerance, the space-group median and the non-mergeable control | two tolerances and four results in one long sentence | `SPLIT`: the strict result and its reason; why 0.1 is used; the restoration of the bound; the space-group increase; the non-mergeable control | shows the outcome of the intervention |
| S17-47 | `1603-1606`, the generative inference and the satisfying entries | the primary inference and the internal control are clear | `KEEP both sentences` | closes the generative sample |
| S17-48 | `1607-1615`, the experimental control sample, the 27 cases, the 0/27, the examples, why the sites differ and the final inference | the control design, the result, the chemical examples and the interpretation make one over-long paragraph | `SPLIT/BREAK`: the control cohort; the 27 eligible and 0 restored; examples of genuine ordering; the comparative inference | strengthens the generative-versus-experimental contrast |
| S17-49 | `1616-1620`, the database lookup, the 2/300, the novelty criterion and why merging is direct | the complementary test and the interpretation share a sentence | `SPLIT`: the lookup method and result; the relation to the novelty criterion; why the intervention is a direct check | completes the check on the alternative explanation |
| S17-50 | `1624-1628`, the second protocol, the Law 8 audit mismatch, the recomputation and the preservation | the protocol, the error, the correction and the archive share a paragraph | `SPLIT`: the pre-fixed extension protocol; the audit finding; the recomputed definition; the preserved original artefacts | records the implementation correction clearly |
| S17-51 | `1628-1631`, the integer and fallback assignment, and no verdict | the two assignment steps and the outcome share a paragraph | `SPLIT`: the integer attempt; the mean-valence fallback; neither -> no verdict | defines the charge pipeline |
| S17-52 | `1632-1635`, the 7.5% subset, the intermetallic remainder, the per-law satisfaction and the expected reason | the coverage and the conditional performance share a paragraph | `SPLIT`: the assignable coverage and why it is low; the per-law rates; the within-bound interpretation | separates applicability from performance |
| S17-53 | `1636-1638`, the Set 3 against seven-law rates, the 335/314 denominators and the 159/62 outcomes | two sets and several outcomes in one sentence | `REWRITE` into a small table: set, evaluable n, satisfied/failed/no verdict | prevents the denominators being misread |
| S17-54 | `1638-1641`, the 155 failures divided into four Law 7/8 classes | a long string of four counts | `REWRITE` into a parallel list or small table, making the total of the four classes visible | shows the attribution of the added laws |
| S17-55 | `1641-1643`, the Law 8 coverage and the 36/254 | the coverage limit and the failure rate share a group of sentences | `SPLIT`: the coverage and limiting factor; the conditional result | evaluability first, then the verdict |
| S17-56 | `1643-1646`, the Law 7 26.1% against 40.6%, the subset symmetry and the different denominators | the comparison, the reason and the denominator note are too dense | `SPLIT`: the two rates; the subset chemistry and symmetry reason; the denominator warning | explains what look like inconsistent numbers |

### The seven-generator benchmark and the coordinate checks, lines 1648-1757

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S17-57 | `1650-1655`, the seven models, the deposit and licence, the 500 per model and the MP-20 control definition | the benchmark provenance, the sample size and the control parenthesis make one over-long sentence | `SPLIT`: the dataset, models and licence; the per-model sample; the definition of the experimental control | establishes the external benchmark |
| S17-58 | `1655-1656`, distance filters do not separate them, the law sequence does | the contrast is short and forceful | `KEEP` | enters the PRIS result |
| S17-59 | `1657-1663`, the charge no-verdict rates, the three generator groups, the Set 4 rates and the control | the applicability, the three model classes and the satisfaction rates are too dense | `SPLIT`: the charge coverage; the coordinate-only models; the symmetry-constrained models; the experimental control | presents the separation by mode of generation |
| S17-60 | `1663-1664`, other unavailable measurements and no verdict | the two sentences can be merged | `MERGE` into one denominator rule | avoids repeating the no-verdict definition |
| S17-61 | `1665-1667`, the strict Law 7 rates and the archive | the result and the availability are mixed | `SPLIT`: the Law 7 outcome; the archive statement | closes the raw-output analysis |
| S17-62 | `1669-1672`, `Two objections... The first...` | the metadiscourse is slightly long | `REWRITE` into a direct numbered structure, `First, ...`; the opening sentence states only that there are two measured checks | gets to the robustness quickly |
| S17-63 | `1672-1677`, the full-denominator Law 7 rates, the same separation and the charge explanation | the test, the three rates and the inference make one group of sentences | `SPLIT`: why Law 7 uses the full cohort; the no-symmetry, symmetry and control rates; what this rules out | completes the first objection |
| S17-64 | `1677-1679`, the second objection, numerical triclinicity and unrelaxed noise | the concern is clear but the sentence is long | `SPLIT`: state the objection; why orbit detection on unrelaxed cells may reflect noise | sets up the relaxation test |
| S17-65 | `1679-1685`, the 2,500-structure relaxation protocol, the 0.1 tolerance and five model and control rates | the setup and all the results share one sentence | `SPLIT`: the five cohorts and the protocol; the rationale for the tolerance; MatterGen/DiffCSP; the control and SymmCD; MiAD | reports the relaxation robustness clearly |
| S17-66 | `1686-1690`, the reduces-not-erases inference and the scope sentence | both sentences are clear | `KEEP` | gives the strong conclusion for the sets tested |
| S17-67 | `1694-1697`, the inputs checkCIF requires, and that it was not run | the required inputs and the decision share a sentence | `SPLIT`: which inputs are unavailable; therefore the established software was not run | states the reason for the choice directly |
| S17-68 | `1697-1704`, the four reimplemented test families | a very long list, each item with its own thresholds | `REWRITE` into a numbered list: missed symmetry, short contacts, voids, floating atoms; one line each giving the threshold and sensitivity | makes the coordinate checks checkable |
| S17-69 | `1704-1708`, the no-alert-code claim, the Law 8 audit, and the recomputation and preservation | the software-naming boundary and an unrelated audit are mixed | `BREAK`: one paragraph for the coordinate-check naming; one for the Law 8 correction | prevents it being read as official checkCIF output |
| S17-70 | `1709-1712`, the primitive-cell convention, the paired comparability and the discovery incomparability | the method and two consequences for comparison share a paragraph | `SPLIT`: the conversion rule; the validity of paired comparison; the non-comparability of absolute rates | marks the cell convention clearly |
| S17-71 | `1714-1718`, the seeded 8,000-structure scan and first 300, the enrichment, and the uniform sample 222/5000 | the sample selection and the prevalence correction run in succession | `SPLIT`: the enriched sampling method; what it cannot estimate; the uniform-sample reference rate | keeps the 300 from being taken as representative |
| S17-72 | `1718-1721`, the perturbations, computed parents as controls, and why not experimental ones | the operator and the choice of control share a paragraph | `SPLIT`: the damage generation; the control population; the licence reason | defines the paired benchmark |
| S17-73 | `1721-1724`, the parent failure rates and the added paired analysis | the initial result and a post-result change of design run in succession | `SPLIT`: the unpaired parent rates; when the paired analysis was added; the definition of paired detection | transparent in time and precise in definition |
| S17-74 | `1725-1727`, the six-law set definition and the Law 7/8 consistency counts | the method label and the consistency result share a paragraph | `SPLIT`: the composition of the law set; the sample rates; the reference-subset rates | checks the consistency of the cohort |
| S17-75 | `1747-1751`, the pre-specified comparison 0.365/0.021 and the added paired 0.430/0.042 | the conditions and numbers of the two analyses are crowded together | `SPLIT`: the primary restriction and results; the post-result paired restriction and results | keeps the two analyses strictly apart |
| S17-76 | `1751-1755`, the two symmetry tests, the coordinate-check definition and limitation, and the Law 7 question | the contrast is clear but the four sentences could be more parallel | `REWRITE` as `coordinate check asks...; Law 7 asks...`, placing the wrong-site result after the former | makes the difference between the two tests visible at a glance |
| S17-77 | `1756-1757`, the archive inventory | clear | `KEEP` | points at the per-structure evidence |

### Historical falsified depositions, relaxation and the composition screen, lines 1759-1839

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S17-78 | `1762-1765`, why a historical cohort, no refit, and the source | the three sentences are clear | `KEEP` | moves from controlled damage to real cases |
| S17-79 | `1767-1776`, the 184 reports and seven notices, the general fraud pattern, the missing parent provenance and no inference | the corpus size, the history of the mechanism and the provenance restriction share a paragraph | `BREAK`: the archive scale; the reported pattern; the missing lineage; what is not inferred | establishes the boundary of the historical evidence |
| S17-80 | `1776-1783`, the retrieval source, the 177 parsed, the molecular domain, the 172 no verdicts, one control too large, and five evaluable all failing | the retrieval, the parse, the domain and the verdict pipeline are too dense | `SPLIT`: where the CIFs were retrieved; the parse count; the composition of the domain; the 172 no verdicts; the status of the control; the result for the five evaluable | moves from archive to evaluable cohort |
| S17-81 | `1783-1786`, the four M(C2H2N3Cl) entries, the four metals and the no-common-parent inference | the grouping of the sample and the provenance boundary share a group of sentences | `SPLIT`: the four related-framework records; the metal labels; no common-parent claim | states the observed grouping precisely |
| S17-82 | `1786-1791`, the two common violations, the rho value and its reason, the BVS range and the mechanism match | the shared conclusion, two quantities and the mechanistic explanation run in succession | `SPLIT`: the common failures; the contact result and its physical meaning; the bond-valence result; the connection to the cation-swap mechanism | shows the fixed-coordinate diagnosis |
| S17-83 | `1791-1793`, the fifth entry and the archive | the result and the availability share a sentence | `SPLIT`: the KNOF2 result; the archive statement | completes the fifth case |
| S17-84 | `1797-1799`, the MatterSim settings, the convergence and the cap | five settings inside a parenthesis | `SPLIT`: the model, cell and optimiser; the force and cap; the convergence status | defines the common relaxation protocol |
| S17-85 | `1799-1806`, the 150 GNoME parents, the damage counterparts, the parent median, three damage medians and the swap-n limitation | the sample, the baseline, the damage results and the missing-charge limitation share a paragraph | `SPLIT`: the sample and seed; the parent response; compression/expansion/displacement; the sample limitation of the swap class | gives the severity scale clearly |
| S17-86 | `1807-1811`, the MatterGen addendum, the 162 assignable result and the 338 no-charge result | the timing, two cohorts and several statistics share a paragraph | `SPLIT`: the pre-fixed addendum; the assignable cohort; the no-charge cohort | compares the raw outputs in parallel |
| S17-87 | `1812-1815`, what the sample cannot measure, what it shows, the separate experiment and the repeated protocol | `protocol fixed` repeats at the beginning and the end, and `only` is explained twice | delete the repetition in the final sentence; `SPLIT` into the limitation, the supported result and the relation to the damage experiment | tightens the interpretation |
| S17-88 | `1820-1826`, why compare with a composition check, the validity lineage and the two component tests | the motivation and the provenance of the metric share a paragraph | `SPLIT`: why it is included; the metric and its source; the composition and distance components | defines the incumbent baseline |
| S17-89 | `1826-1829`, the smact settings, the 76.8% and the stricter-chemistry comparison | the protocol, the result and the interpretation share a sentence | `SPLIT`: the implementation; the satisfaction; what it tests | describes the selectivity of the composition screen |
| S17-90 | `1829-1833`, structurally zero, composition preserved, the original verdict, the conditional 0.000 and the repeated conclusion | the null result is repeated in four ways | `COMPRESS`: keep the measured 0.000 across all types and the composition-preserving mechanism; delete the repeated "nothing to measure" phrasings | gives the exact zero and its cause concisely |
| S17-91 | `1833-1834`, the archive | clear | `KEEP` | points at the source data |
| S17-92 | `1836-1839`, the trade-off paragraph | three logically clear sentences with a forceful conclusion | `KEEP`; only smooth the line break and the conjunctions of the last sentence | preserves the clear use-case decision |

## B18. Supplementary Note S18: robustness, equations and implementation

### Reduced contact and the exact trees, lines 1841-1879

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S18-01 | `1843-1844`, the Note S18 navigation sentence | clear | `KEEP` | states that this section collects the quantitative detail |
| S18-02 | `1848-1851`, the tail shape, the quadratic and log density, the RSS and curvature, and the repulsive-wall interpretation | the fit setup, the numbers and the physical meaning share a sentence | `SPLIT`: the distribution and fitted range; the linear against quadratic residuals and the curvature; the meaning of the repulsive tail | moves from the statistical shape to the physical interpretation |
| S18-03 | `1851-1854`, the 0.735 percentile, that it is not a constant, and the four-point sweep | the definition and the list of thresholds share a sentence | `SPLIT`: the provenance of the first percentile; the absence of a sharp transition; the four sweep values as a parallel list | makes clear the parameter is tunable and not a physical constant |
| S18-04 | `1854-1857`, the anion-stratified satisfaction and the phosphide damage result | the family-wide stability and the exception share a sentence | `SPLIT`: the satisfaction across the ten families; the phosphide damage detection and control count | foregrounds the chemistry-specific response |
| S18-05 | `1857-1859`, `criterion is not wrong ... almost no detection ... satisfaction alone` | phrased around "not wrong", explaining the same difference three times | `REWRITE` into a direct sentence: for phosphides the law preserves satisfaction but contributes little damage detection | states both metrics in one sentence |
| S18-06 | `1859-1860`, the DFT-relaxed satisfaction | clear | `KEEP` | rules out inflation by relaxation |
| S18-07 | `1864-1867`, the tree against the core, the cost-ratio plateau, the F1 invariance and only the cost moving the curve | the performance comparison and the objective robustness share a paragraph | `SPLIT`: the tree and core point; the ratio plateau; the objective invariance; the trade-off controller | delimits the operating point clearly |
| S18-08 | `1868-1869`, the depth-4 runtime with no proof, and the scope of exactness | both sentences are clear | `KEEP` | limits optimality to depth <= 3 |
| S18-09 | `1869-1871`, the omitted-type mean drop and the single-threshold comparison | two sets of seen and omitted numbers share a sentence | `SPLIT`: the tree, seen -> omitted; the single threshold, omitted; the comparison | foregrounds the omitted-class sensitivity |
| S18-10 | `1871-1874`, the five signed changes, the cancellation and the mean absolute change | a list of five and the summary share a sentence | `SPLIT`: the signed changes; why the mean cancels; the absolute-change comparison | explains the choice of metric |
| S18-11 | `1874-1876`, the tree winning 3/5, and the aggregate against per-type disagreement | clear | `KEEP` | reports both the aggregate and the per-type picture |
| S18-12 | `1876-1879`, the selected features and the damage-type pattern | the feature list and the inference share a sentence | `SPLIT`: which features are selected; their per-type pattern | shows that the tree mechanism differs |

### Cross-database ranking, ties and equations, lines 1881-1958

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S18-13 | `1883-1888`, the four energy spans, the order-of-magnitude meaning, the four matched-window accuracies and the ranking reversal | two sets of four numbers and the explanation share a paragraph | `SPLIT`: the energy spans; the task-difficulty interpretation; the matched window; the four accuracies and the ranking reversal | separates the distribution shift from the matched result |
| S18-14 | `1889-1895`, the four largest-group shares, the pair/group weighting reversal and the reporting principle | composition dominance and the weighting result share a paragraph | `SPLIT`: the largest shares; the pair-weighted result; the group-equal result; the implication for reporting | explains where R9 came from |
| S18-15 | `1896-1900`, the descriptive full-set PSS, the three exclusion conditions and the unchanged conclusion | the exploratory status and three equal scores share a paragraph | `SPLIT`: the status of the analysis; the full, exclude-SiO2 and exclude-framework values; the unchanged inference | reports the composition robustness clearly |
| S18-16 | `1900-1903`, the volume score, the denser proportion and the persistence in non-oxides | three supporting results in succession | `SPLIT`: the volume baseline; the non-silica density direction; the non-oxide result | supports the packing interpretation |
| S18-17 | `1905-1909`, the high-pressure exclusion check | clearly expressed | `KEEP` | rules out the density term being driven by high-pressure phases |
| S18-18 | `1913-1917`, the decision rates, group tie rates, conditional accuracies and 39-225 groups of the four law sets | three metrics across four sets stacked into two sentences | `REWRITE` into a small table: set, pair decision, no-unique-group, conditional accuracy, contributing groups | shows coverage and accuracy together |
| S18-19 | `1917-1920`, ties being appropriate, thresholding discarding ordering, and rho being continuous | the three sentences are logically clear | `KEEP` | explains the filter/ranking distinction |
| S18-20 | `1924-1928`, `The eight PRIS laws, written as applied...` | the opening is a sentence fragment, and the definitions of rho, f_i and EM are crowded into the following sentence | `REWRITE` into a complete lead-in; a short sentence per symbol; separate out the Ewald and model-bound note | introduces the equations clearly |
| S18-21 | `1929-1941`, the Law 1-8 equations | the mathematics is already the clearest form | `KEEP` | changes no threshold or definition |
| S18-22 | `1943-1945`, the standardisation and the missing-value imputation | two preprocessing steps in one subordinate clause | `SPLIT`: define the standardised value; define the frozen-median (x^\dagger) | separates scaling from missing-value handling |
| S18-23 | `1945-1947`, the positive difference and the introduction of the zero-intercept score | clear | `KEEP` | explains the direction of the score |
| S18-24 | `1954-1958`, the three descriptor definitions and the two topology definitions | the formula sentences are clear | `KEEP`; the two topology items may be written in parallel | completes the PSS glossary |

### Phonons and the energy/order distinction, lines 1960-2107

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S18-25 | `1962-1965`, the 26,600 cohort, the two phonon sources with their counts and framework, and the pre-fixed protocol | the cohort definition and the provenance are too dense | `SPLIT`: the total cohort; the DFPT source and count; the finite-displacement source and count; the protocol | establishes the phonon population |
| S18-26 | `1965-1969`, the three properties, and the two imaginary-mode definitions with their threshold sensitivity | three analysis variables and a source-specific cutoff in one long sentence | `REWRITE` into three numbered items, with the imaginary-mode item subdivided into the DFPT and DOS definitions | makes the categories reproducible |
| S18-27 | `1969-1971`, the cross-method and the band-against-DOS agreement | the two agreements have different denominators | `SPLIT` into two sentences, naming the 1,175-structure common cohort | prevents 88.8% and 99.1% being conflated |
| S18-28 | `1971-1974`, the three main findings | three headline findings in one sentence | `SPLIT` into three sentences: on-hull imaginary modes; recorded metastability; unrecorded stable structures | keeps each concept separate |
| S18-29 | `1975`, `Satisfaction of each law is the source...` | awkward syntax and an unclear back-reference | `REWRITE` into direct navigation: the per-law satisfaction analysis underlying the distinct-site comparison appears in S18.7 | points forward explicitly |
| S18-30 | `1975-1982`, the two limitations and the lenient-tolerance result | the sampling bias, the tolerance sensitivity and two population shifts share a paragraph | `SPLIT`: the sampling bias; the finite-displacement sensitivity; the experimental shift; the theoretical shift; the comparative conclusion | separates the limitations from the robust contrast |
| S18-31 | `1986-1987`, `two conditions on this population` | `two conditions` are not named | `REWRITE` to write out which two of the experimental record, hull energy and imaginary modes are actually compared | removes the vague back-reference |
| S18-32 | `1987-1989`, all five being evaluable because the stored relaxed structures were not downloaded again | "not downloaded again" is an implementation detail interrupting a scientific definition | `COMPRESS` to: all law quantities were computed on the same stored relaxed structures | keeps the comparability and drops the operational trivia |
| S18-33 | `1989-1994`, the charge convention, the requirements, the no-verdict denominator and the five subset sizes | the charge algorithm and five denominators share a group of sentences | `SPLIT`: the charge assignment; the ionic requirement; the per-set denominator rule; the five subset counts as a table or parallel list | defines the evaluable cohorts |
| S18-34 | `1995-1996`, the 14,986-structure common cohort, the four-set sequence and Set 1' separately | clear | `KEEP` | states the population of the table comparison |
| S18-35 | `2018-2021`, the 3,929 failures and the per-law counts | a long string of six law counts | `REWRITE` into a small table or a semicolon list, noting the multi-law failures | gives the mechanism frequencies readably |
| S18-36 | `2021-2029`, the dominant mechanisms, the Law 7 all-against-BVS-subset rates, the subset bias and the denominator principle | the result, the population effect and the reporting rule make one over-long paragraph | `SPLIT`: the dominant contributors; the full Law 7 rate; the joint-coverage rate; the composition of the subset; the same-population rule | explains the dependence on the denominator |
| S18-37 | `2029-2032`, the introduction of two robustness checks and the neighbour-alternative result | it says `Two` but the second arrives much later, so the structure is unclear | `REWRITE` into a `First` paragraph: the primary Law 6 convention, the alternative shell, and the Set 3/4 changes | marks robustness check 1 clearly |
| S18-38 | `2033-2037`, the molecular-anion chemistry and its treatment | the example list and the conclusion are clear | `KEEP`, opening with `Second` | marks robustness check 2 clearly |
| S18-39 | `2038-2041`, the implementation audit, the old radial definitions, the recomputation and the old columns | the error, the correction and the status share a paragraph | `SPLIT`: the audit finding; the production recomputation; the role of the earlier columns | records the repair clearly |
| S18-40 | `2043-2049`, the sampling limitation, the benchmark against application missing convention, the comparison advice and the source tables | two limitations and the navigation run in succession | `SPLIT`: the sampling limitation; the difference in missing-input convention; the implication for comparison; the data availability | closes the interpretation of the table |
| S18-41 | Fig. S21, `2054-2060` | the axes and class crossing, two laws, two record statuses and the lines share a paragraph | `SPLIT`: the cohort and axes; the upper and lower profiles; the markers and lines | layers the caption by graphical element |
| S18-42 | Fig. S22, `2067-2072`, the merge-group ordering and relaxation, the points, the bar, the sorting and the 300 K line | the setup and four visual encodings run in succession | `SPLIT`: the enumeration and DFT setup; the points; the horizontal bar and sorting; the temperature line | the reader understands the estimator first |
| S18-43 | Fig. S22, `2072-2077`, the two medians and 358x, the 0 against 18/23, the solid-solution inference and the note navigation | two sets of results, the thermodynamic interpretation and the navigation share a paragraph | `SPLIT`: the GNoME and control energies; the factor; the 300 K counts; the physical interpretation; the pointer to the Note | foregrounds the ordering result |
| S18-44 | Fig. S23 panel a, `2084-2091`, the 200-cell composition, the five operators, the 50 parents and 20 GNoME, the same protocol, the correlation and the floor | the cohort, the methods and the result are too dense | `SPLIT`: the total cohort; the damaged component; the undamaged component; the paired calculation; the correlation; the floor points | defines the paired DFT validation clearly |
| S18-45 | Fig. S23 panel b, `2091-2093` | the purpose and the bar encoding share a sentence | `SPLIT`: the class-resolved purpose; the filled/hatched mapping | makes the caption easier to scan |
| S18-46 | Fig. S24, `2099-2105` | the flow and the no-verdict convention are clear | `KEEP` | shows the charge coverage |

### Threshold stability and timing, lines 2109-2159

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S18-47 | `2111-2115`, the bound definition, the same-percentile rederivation, the movement of the value and the verdict flips | the definition and a three-step stability procedure share a paragraph | `SPLIT`: the provenance of the percentile; rederive on the held-out set; measure the threshold shift; measure the verdict flips | states the stability test clearly |
| S18-48 | `2136-2137`, the contact, packing and valence shifts, and Law 8 on the same grid | clear | `KEEP` | summarises the stable bounds |
| S18-49 | `2137-2141`, the electrostatic tail, the 15 -> 4 eV, the <0.4% flips and the interpretation | the threshold shift and the verdict stability share a group of sentences | `SPLIT`: why the tails are sparse; the change of threshold; the experimental flip rate; the strong verdict-stability conclusion | separates the numbers from the classifications |
| S18-50 | `2141-2142`, the discrete grid of Law 7 | clear | `KEEP` | explains the 2/3 -> 3/4 |
| S18-51 | `2146-2149`, the benchmark setup and the three cell sizes for Law 1 | the setup and the timings share a paragraph | `SPLIT`: the hardware and process protocol; the Law 1 operation; the three timings | establishes the microbenchmark |
| S18-52 | `2150-2154`, the operations and timings of the full implementation, the charge cost and the cached estimate | the current cost and the potential optimisation share a paragraph | `SPLIT`: the eight-law implementation; the three timings; the charge overhead; the cached-neighbour estimate | separates what was measured from what is projected |
| S18-53 | `2154-2158`, the law timings on a 10k queue and the DFT projection | two queue estimates and a literature projection share a paragraph | `SPLIT`: the contact queue; the eight-law queue; the per-structure DFT assumption; the 10k projection | builds the cost hierarchy |
| S18-54 | `2158-2159`, the order-of-magnitude disclaimer | clear | `KEEP` | makes clear the DFT number is a projection |

## B19. Supplementary Note S19: the shared evaluation procedure

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| S19-01 | `2163-2166`, the three raw split counts and the definition of the analysis subset | the totals and the analysable subset share a paragraph | `SPLIT`: the raw partition counts; the charge and feature eligibility | separates the assigned from the analysed population |
| S19-02 | `2166-2170`, the structure-ID split, the same composition across partitions, what consistency measures and the omitted-type evidence | the split unit and two meanings of validation run in succession | `SPLIT`: the structure-level hashing; the composition overlap; the interpretation of split consistency; the omitted-class evidence | delimits what a random split means |
| S19-03 | `2172-2174`, the PSS preregistration and hash, and the model form | governance and the statistical definition share a paragraph | `SPLIT`: the protocol lock and archive; the antisymmetric zero-intercept model | the plan first, then the model |
| S19-04 | `2174-2178`, imputation, standardisation, pair weighting, forward selection, CV, two stopping rules and the eight-term cap | seven training rules crowded together | `REWRITE` into an ordered procedure: preprocess -> weight -> select/CV -> stopping/cap | makes the PSS fitting reproducible |
| S19-05 | `2179-2181`, the 1,508 split counts, the seeded hash and the script refusing a repeat | the cohort split and the evaluation gate share a sentence | `SPLIT`: the 919/589 assignment; the one-shot held-out gate | separates the data split from its enforcement |
| S19-06 | `2181-2184`, the full-pool transfer and the archive inventory | the application and the reproducibility share a paragraph | `SPLIT`: the no-refit transfer; the archived coefficients, statistics, verdicts and logs | closes the PSS protocol |
| S19-07 | `2186-2188`, the Set 4 preregistration, the feature timing, the Set 3 reproduction and the damage recreation | four gates in one sentence | `SPLIT`: the timing of the plan; the Set 3 reproduction; the damage recreation | establishes the prerequisites of the selection |
| S19-08 | `2188-2191`, the greedy objective, the satisfaction floor, the percentile grid and the one-shot held-out evaluation | the fitting rules and the evaluation share a paragraph | `SPLIT`: the objective and constraint; the grid; the held-out evaluation | separates selection from test |
| S19-09 | `2191-2194`, the omitted variants and the PSS confidence breakdown | the Set 4 transfer rule and the exploratory PSS analysis do not belong to the same topic | `BREAK`: the Set 4 LOFO rule; the post-evaluation PSS confidence analysis | avoids jumping between analyses |

## B20. The DFT supplement: four learned quantities, lines 2196-2258

| no. | line and quoted phrase | problem | specific action | function after the change |
|---|---|---|---|---|
| DFT-01 | `2198-2201`, the four learned quantities | the four comprise bounds, a claim, a severity scale and a property target, and are not grammatically parallel | `REWRITE` into a numbered or semicolon list of four uniform noun phrases | gives a clear map of the DFT campaign |
| DFT-02 | `2201-2204`, each rederived, the pre-fixed protocol and the repository artefacts | the methodological principle and the archive path are clear | `SPLIT` into the pre-fixed DFT rule and the artefact location | separates the scientific protocol from availability |
| DFT-03 | `2206-2210`, `VASP 6.3.0 with...` | the paragraph opens with a predicate-free fragment, and the code, potentials, cutoff, mesh, EDIFF and smearing are all strung together | `REWRITE` into a complete sentence; split into one sentence for the code, potentials and cutoff and one for the k-mesh, convergence and smearing | fixes the grammar and lowers the parameter density |
| DFT-04 | `2210-2212`, the cell relaxation, the constant-volume ISIF and the 1,917 tasks / 344 node-hours | two task settings and the campaign scale run in succession | `SPLIT`: the relaxation settings; the energy-volume settings; the campaign size | separates the type of calculation from its scale |
| DFT-05 | `2214-2218`, the four items listed after `Five experiments answer four questions` | the contact landscape and its control count as two experiments, so the reader has to work out the 5 -> 4 mapping | `REWRITE` into an explicit numbered mapping: Q1 includes two experiments; Q2/Q3/Q4 one each | makes the experiment-question correspondence explicit |
| DFT-06 | `2218-2219`, `Two of them need their estimator stated... first one that comes to hand` | colloquial metadiscourse, and the two are not named | `REWRITE` to say directly that the ordering and property tests require distinct evaluation quantities | moves into the two estimators |
| DFT-07 | `2221-2224`, the released ordering above the best, the 13/23 at the minimum, and what it does and does not measure | the distinction between metrics and the positive result share a group of sentences | `SPLIT`: the definition of the ground-state metric; the 13/23 result; why this metric does not settle disorder | separates ground state from thermal order clearly |
| DFT-08 | `2224-2227`, the random-ordering cost, the 0 against 18/23 at 300 K, and the figure | the estimator, the result and the navigation share a sentence | `SPLIT`: the order-disorder estimator; the control and GNoME counts; the figure pointer | foregrounds the thermodynamic test |
| DFT-09 | `2229-2232`, the 400 GPa proxy target, the 0.940 factor, the consequence for the absolute threshold and the 0.7% | the calibration and the failure of the absolute scale share a paragraph | `SPLIT`: the target scale; the median calibration factor; why the raw threshold is invalid; the 0.7% illustration | explains why a rescaling is needed |
| DFT-10 | `2233-2235`, the r=0.769, the ranking transfer and the rescaled selection | the correlation, the object of transfer and the result share a sentence | `SPLIT`: the correlation; the transfer of ranking and not absolute values; the rescaled selection outcome | makes clear what is being validated |
| DFT-11 | `2237-2241`, the 260 CIFs, one per candidate, and the five index fields | the data availability and a long list of fields share a sentence | `SPLIT`: the CIF availability; the before and after fields of the index; the PSS and modulus fields | describes the Supplementary Data clearly |
| DFT-12 | `2241-2242`, `Two things ... worth stating...` | metadiscourse | `CUT` and go straight to the composition result | removes a redundant transition |
| DFT-13 | `2242-2244`, the 11 elements and the dominance of Os/Ir/Re | clear | `KEEP` | reports the concentration in the periodic table |
| DFT-14 | `2244-2246`, the sentence opening with `And`, and the Law 7 symmetry contrast | colloquial linkage and a convoluted subject | `REWRITE` into a direct sentence: relaxation changes the symmetry used to assess Law 7 | moves into the before/after result |
| DFT-15 | `2246-2248`, the 61 -> 113, none reversing, and the merging explanation | the numbers, the direction and the mechanism share a sentence | `SPLIT`: the before and after counts; no reverse moves; the site-merging mechanism | explains the effect of relaxation clearly |
| DFT-16 | `2248-2252`, the 57.9% against 11.7%, the unrelaxed score anticipating it, and the construction being unable to arrange it | the subgroup result and the independence conclusion share a paragraph | `SPLIT`: the retained-against-removed comparison; the predictive relation; the independence of the check | closes the property-screen validation forcefully |
| DFT-17 | `2254-2258`, the four-parameter five-point fit, the leave-one-out shift and the three-orders comparison | the degrees of freedom of the fit, the sensitivity test and the magnitude share a paragraph | `SPLIT`: the one degree of freedom; the drop-one procedure; the median shift; the relative magnitude | reports the stability of the modulus fit clearly |

# C. Paragraph coverage across the whole text

This table exists so that the implementation is not "a few typical examples changed". `Itemised` means every sentence in that paragraph needing change is located in the tables above; `KEEP` means every sentence was read and none was found wordy, hard to follow or incoherent.

| file | coverage | status | corresponding numbers |
|---|---|---|---|
| `main.tex` | the active author/front/body/methods includes, plus acknowledgements and data/code availability | itemised or explicitly kept | M-01-M-06 |
| `front_meta.tex` | every sentence of the title and abstract | itemised | FM-01-FM-12 |
| `front_body.tex` | every paragraph of the Introduction | itemised or kept sentence by sentence | I-01-I-37 |
| `body.tex` | every paragraph of Results 2.1-2.5 and the Discussion, and every figure caption | itemised or kept sentence by sentence | R-H1-R-H5; R-01-R-141 |
| `methods.tex` | every paragraph of the opening and the five subsections | itemised or explicitly kept | ME-01-ME-68 |
| `si.tex` | the SI title, abstract and wrapper text | itemised or explicitly kept | SW-01-SW-04 |
| `si_body.tex` | every prose paragraph, caption and table-introduction sentence of S1-S19 and the DFT note | itemised or explicitly kept | S1-01-S19-09; PU-01-PU-33; DFT-01-DFT-17 |
| `si_body.tex` | pure formulas, purely numerical table cells and LaTeX layout commands | kept | no linguistic change |

# D. Order of implementation, once the user confirms

1. Revise `front_meta.tex` and `front_body.tex` first, fixing the narrative spine of the title, the abstract and the Introduction.
2. Then revise `body.tex` per R-01-R-141; every headline result, number and strong conclusion stays as it is, and only sentences are split, duplication removed, references repaired and paragraphs rearranged.
3. Revise `methods.tex` per ME-01-ME-68, prioritising clarity on the splits, the denominators, no verdict, the PSS fitting and the DFT protocol.
4. Synchronise the `si.tex` title and revise `si_body.tex` per B1-B20; any technical detail that must survive a compression in the main text stays in the SI.
5. Run one terminology-consistency pass: PRIS/PSS, satisfaction/damage detection, fail/not satisfy/no verdict, discovery/held-out/reserve, deposited/as-given/primitive/relaxed cell.
6. Compile the main text and the SI and check the cross-references, table and caption overflow, pagination and title synchronisation; fix only what these linguistic changes caused.
7. Deliver a per-file diff, with a "original sentence -> revised sentence -> reason" checklist so the authors can accept or revert each item.

# E. Immutable constraints during implementation

- Do not weaken any conclusion, and do not automatically soften strong conclusions such as `proves`, `can` or `shows`.
- Do not delete a headline number, and do not change any value, formula, sample size, threshold, citation or figure conclusion.
- Do not change the paper's existing order of findings: autonomous discovery -> laws/mechanisms -> benchmark performance -> PSS/synthesizability -> inverse design/external diagnosis -> broader significance.
- Do not review or modify any file outside the scope of this plan.
- Do not modify any of the paper source files above until the user explicitly confirms.
