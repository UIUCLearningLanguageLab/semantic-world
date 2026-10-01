# Proposal: decisions for stage 3 of the corpus generator

October 1, 2026. Raised at the end of stage 2 of `docs/specs/CORPUS_GENERATOR.md`, and answered by Jon before stage 3 (scenes and events). Status: decided. The specification was updated to match, as decisions 27 to 31.

## Jon's decisions

### 27. Joined relative clauses

Stage 2 skipped a rule term that needed more than one relative clause. In the default world that left 27 of 103 terms unstated.

**Decision.** A relative clause can join several verb phrases with "and": "things with wings that are not red and not big can fly". All of a rule term's negated IS literals go into one relative clause. The stage 2 code now counts those terms as stateable. The grammar itself is built in stage 4. Terms that read a scalar threshold stay unstated.

**Result.** The default world now states 86 of its 103 rule terms, up from 59. Of the other 17, 13 read a scalar threshold, and 4 have no instance that satisfies them. All 27 terms that were skipped for their relative clauses are now stated. The tiny world states 19 of 26, up from 16.

### 28. Law-like test items

Under the law-like reading, "all penguins swim" is false when every penguin in the world swims but nothing fixes it. In the default world, 1,158 of the 2,278 pairs of a category and a feature on which every instance agrees are not fixed.

**Decision.** A false `all` or `no` item that no instance contradicts is marked. Such items go into a test set of their own. Ordinary test sets then hold only items that observing the instances could decide. Built in stage 6.

### 29. False event items

**Decision.** A false event-level item is an event that did not happen in the scene. Each item is marked "possible" (the world allows the event) or "impossible" (the world rules it out), and the two kinds go into separate test sets. Built in stage 6.

### 30. Test context

**Decision.** Every instance-level and event-level test item names a document. The item is tested as a continuation of that document, so its definite noun phrases refer to that document's referents. Built in stage 6.

### 31. Quantifiers in rule statements

The specification rendered the bare plural "things with wings and with feathers can fly" as `ALL(...)`.

**Decision.** A rule statement's quantifier is drawn like that of any class-level proposition: `all` or the generic, at the generic rate. The logical form then matches the surface: "all things with wings ..." is `ALL`, and the bare plural is `GEN`. A rule statement is true under both.

## What stage 3 builds for decisions 28 to 30

Stage 6 builds the test sets. Stage 3 builds what they need from scenes:

- An event-level proposition can name no event. It is then true when such an event occurred at any time step of its scene, and false otherwise. So "an event that did not happen in the scene" is a judgment the truth tests already make.
- The grounding of every event-level proposition holds `possible`: whether the agent has the CAN feature, or the verb's relation holds for the agent and the patient. A true event is always possible. A false one is marked possible or impossible by the same field.
- An event-level proposition is valid only when its agent and its patient take part in its scene. A test item is a continuation of a document, so its referents are that document's participants.

## Choices made by Claude

These are engineering choices made while building stage 3. Each one is the working design unless Jon changes it.

1. **Scenes do not depend on the lexicon.** The generator uses every CAN feature and every verb, with a word or without one. The planner leaves out an event that no word can report. So changing the lexicon settings never changes a scene.
2. **One part of the stream for each scene.** Scene `SN.<n>` draws from `corpus:scenes:SN.<n>`. A scene depends only on the corpus seed, its number, its seed instance, and the scene settings.
3. **Participants.** The other participants are drawn without replacement, and the seed is never drawn again. An instance whose weighted sum is 0 is never drawn, so a scene can be smaller than `scene.size`. The participants are listed with the seed first, then in the order they were drawn.
4. **Taxonomic similarity.** The similarity of two leaves is computed as in `thematic.csv`: the configured metric over the leaves' generative vectors. `thematic.csv` lists only the pairs with a thematic score above 0, so the generator computes the similarity of every pair itself. An undefined or negative similarity counts as 0.
5. **The two weights of an event.** An event is transitive with probability `scene.transitive_share` when both kinds of event are possible, and of the only possible kind otherwise. A share of 0 or 1 rules the other kind out entirely. Within its kind, an event is drawn with probability proportional to its verb's weight.
6. **`scene.verb_weights`.** The keys are CAN features and verbs (the leaves of the verb tree). A label that is left out has the weight 1. A label that the world does not have, or a verb category, is a configuration error that names the field.
7. **Events use verbs, not verb categories.** A verb category names an event only when the event is mentioned (decision 23). The event keeps its own verb.
8. **Repeats.** The same event does not occur twice at one time step. It can occur again at a later step, because nothing changes state.
9. **Event labels.** Events are numbered within their scene in time order, and within a time step in the order they were drawn.
10. **The event's logical form.** An event-level proposition has the instance-level form, with the level `event`, its scene, and the event it reports. Two reports of two different events are two different propositions, even with the same verb, agent, and patient.
11. **Naming an event's verb.** The candidate names are the event's verb and the verb categories above it that have a word. One is drawn with the weights of `mention.verb_level_weights`. A level with the weight 0 is never used. When no candidate has a positive weight, the event cannot be reported.
12. **`scenes.jsonl`.** A scene is written with `steps` as a list of lists: one list of events for each time step, empty steps included.

## Still open

Nothing new. The stage 6 questions of the earlier proposals are answered by decisions 28 to 30.
