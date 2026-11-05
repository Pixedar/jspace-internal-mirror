# Two ways a model of a process can enter the process

Wiktor Tomasik · external reader snapshot 3 November 2026 · Internal Mirror experiments begun 4–5 November 2026

Put the same passage in front of two copies of Qwen. In the external-reader study, a small attached controller might give different weight to the passage in each copy because its selection has a different history. In Internal Mirror, Qwen itself can interpret a description of its activity and choose how to change the next step. Both create feedback. They differ in what carries the self-description and in how it gets a say.

| | External attention reader | Internal Mirror |
|---|---|---|
| What is measured | Recent Qwen activations and what the attached reader selected | Qwen activity while writing, translated into short lists of candidate words |
| Who interprets it | A learned controller predicts what its own next selection will emphasize | Qwen interprets the list and can describe what the measurements show |
| How it acts | The prediction helps set the reader's next selection. In live mode, a writer feeds selected native content back into Qwen. | A readout or conclusion enters Qwen's context. Tool choices can add activation biases or concept-to-concept rules. |
| What stood out | Different selection habits survived even when the compact pictures were mostly alike. | Qwen could forecast a rule correctly, keep choosing the opposite, then act on its conclusion when the harness put it at the choice. |

## A way of looking can persist without a distinctive picture

In the external-reader recordings, copies developed distinct selection habits. Much of the compact picture, however, followed the category of the text they were all reading. In one displayed start, copy identity accounted for 95% of variation on the displayed stance axis, while text category accounted for 92% of the picture axis. These are descriptive axes fitted for that start. The broader control result is that a coarse or fixed picture can preserve substantial selection organization, while weakening responses to changes in the text. It separates *what gets picked out* from *the picture formed from it*. [External reader study](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html#self-models)

In live tests, restoring a specific evidence-related direction inside Qwen removed part of the difference between two controllers' answer preferences. This shows a route by which selection changes how strongly Qwen favors an answer. The effects were usually shifts in preference strength. The swap controls leave ownership of the forecast open as an explanation. [Meaning chapter](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html#meaning)

## A description can be true and still sit outside the choice

In Internal Mirror, the heading “your own workspace” changed how Qwen used the following words even when instructions described the effect. Separately, Qwen learned to predict the effect from outcomes. When its conclusion appeared in the decision prompt, it could change the choice. The 8B and 14B results differed in the format that worked. This is an experimental separation between a default processing response, a report about that response and using the report to act. [Internal Mirror research notes](RESULTS.md)

The external reader puts its prediction directly inside the rule that chooses what to emphasize. Internal Mirror puts an explicit description inside the prompt that chooses what to do. A successful intervention in either setup demonstrates that the information reached a decision through that route. The first setup provides stronger evidence about persistent selection histories. The second makes semantic feedback visible when a word such as *strangers* becomes the subject of an intervention.

For questions about mind, the comparison makes “a system has a self-model” too coarse a description. A system can retain a characteristic way of looking, have an explicit account of what it does, and use that account to alter a decision. These functions can vary separately. Attention schema theory motivates the first question about regulation. Global workspace ideas motivate asking which mechanisms can use the information after it is made available. The measured effects concern selection, word uptake and choice in two instrumented systems. They leave subjective experience and formal claims about integrated information open.
