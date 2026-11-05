# J-space Internal Mirror

**What happens when a language model reads and acts on a measurement of its own activity?**

Wiktor Tomasik · experiments begun 4–5 November 2026

[Read the illustrated study](https://pixedar.github.io/projects/self-awareness/internal-mirror/) · [Explore the measurements](https://huggingface.co/datasets/Pixedar/jspace-internal-mirror-evidence) · [External attention-reader study](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html) · [Citation](CITATION.cff) · [Reserved evidence DOI](https://doi.org/10.5281/zenodo.23989888)

Imagine Qwen writing a story about a winter forest. Between paragraphs an instrument looks at the activity inside Qwen and renders part of it as a short list of words. If the list arrives as ordinary words, some of those words enter the story and then show up in the next measurement. In one run, the forest acquired “a misdemeanour of grief.” If the list arrives with the heading *your own workspace*, the same feedback route mostly stays quiet.

That difference opens up a deeper question. Qwen can learn to predict that the heading weakens a word's influence, then keep choosing as if the heading strengthened it. Under specific conditions, giving Qwen its own conclusion again at the point of choice changes what it does.

## Three findings to start with

- **The “mine” label changes what gets through.** In a seven-part writing task, Qwen3-8B reused 36.8% of the preceding readout words when they were presented plainly, and 0.9% when they were described as its own workspace. Blocking attention to the heading reduced part of the effect in held-out tests. The label was supplied by the instrument.
- **Predicting and choosing came apart.** With feedback, Qwen forecast the direction of the effect correctly and could state it. In three 8B batches, 27 of 36 runs mostly chose the better framing for a strong word in the second half when given their earlier conclusion at decision time. The matched history-only runs yielded 0 of 36. At 14B, the useful route in the tested batch was a fresh numerical forecast.
- **“Strangers” became part of a feedback loop.** One run installed a rule that made *explorer* activate *discovery*. After the writing got caught in that feedback, the readout contained *strangers*. Qwen acted on the word and adopted it in a self-description. It appeared again in eight regenerations from the same partially reconstructed state. Removing the self-written bias or added units lowered its rank in the readout.

![Writing, choice and forecast measurements from the Qwen3 experiments](figures/selftag.png)

## When the rule changed

In a 40-round follow-up, a hidden activation intervention made the own-workspace option stronger. At 14B, choices adapted when Qwen saw its fresh forecast. At 8B, forecasts could follow the change while the written account still described the old rule. [Read the dated follow-up](followups/REVERSAL.md).

## Two routes into a model's next step

The [original J-space project](https://github.com/Pixedar/jspace-strange-loop) builds a small reader alongside frozen Qwen. The reader learns what its own selection tends to emphasize and feeds that prediction into its next selection. In Internal Mirror, Qwen interprets words decoded from its own activity and can choose interventions through an instrument. The first route embeds a learned prediction in a selection rule. The second lets a model interpret and act on a description in language. Both show how much depends on where the description enters the computation.

## Explore the release

| Material | Location |
|---|---|
| Method, individual cases, controls and limits | [Research notes](RESULTS.md) |
| Side-by-side interpretation of the two systems | [Comparison](COMPARISON.md) |
| Run-level measurements, source hashes and CPU check | [Evidence guide](EVIDENCE.md) |
| Experiment code and execution requirements | [Reproduction](REPRODUCE.md) |
| Dated source and evidence record | [Provenance](PROVENANCE.md) |

The experiment uses supplied text frames, a partial readout of activations and interventions installed by the harness at Qwen's request. The results separate effects of a self-label, explicit forecasts and decisions in these tasks. Whether they reflect a broader property of minds or subjective experience is outside what the measurements establish.

## License and attribution

Credit **Wiktor Tomasik** and cite the dated edition in [CITATION.cff](CITATION.cff). Original code is AGPL-3.0-only. Original text, figures and measurements are CC BY-SA 4.0. [License scope](LICENSING.md) · [Agent instructions](AGENTS.md) · [Reuse policy](AI_POLICY.md)
