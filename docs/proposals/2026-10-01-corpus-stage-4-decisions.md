# Proposal: decisions for stage 4 of the corpus generator

October 1, 2026. Raised at the end of stage 3 of `docs/specs/CORPUS_GENERATOR.md` and while building stage 4 (grammar and realization). Status: decided. Decision 32 was decided before stage 4. Jon answered the three questions under "Questions for Jon" before stage 5, as decisions 33 to 35 (`docs/proposals/2026-10-01-corpus-stage-5-decisions.md`): the answers change question A (`grammar.can_rate` replaces `grammar.class_can_rate`) and settle questions B and C as recommended.

## Jon's decision

### 32. Events are drawn verb first

Stage 3 drew each event from the whole pool of possible events, so a verb that holds for many pairs took most of the transitive events.

**Decision.** Events are drawn verb first. `scene.transitive_share` still decides between the intransitive and transitive kinds first. Then a CAN feature or leaf verb is chosen among those with at least one possible event in the scene, weighted by `scene.verb_weights` (uniform by default). Then the participants are chosen uniformly among the pairs, or the agents, for which that verb's event is possible. The stage 3 code and tests were updated.

**Result.** The verb distribution over the same 500 default-world scenes (seed 1), before and after:

| Verb | Events, event first (stage 3) | Events, verb first | Scenes where the verb is possible | Possible pairs per scene |
| --- | --- | --- | --- | --- |
| `V1.1` | 118 | 194 | 224 | 0.89 |
| `V1.2` | 103 | 160 | 206 | 0.80 |
| `V1.3` | 119 | 182 | 207 | 0.74 |
| `V2.1` | 75 | 123 | 157 | 0.50 |
| `V2.2` | 63 | 75 | 137 | 0.41 |
| `V3.1` | 901 | 659 | 461 | 6.34 |
| `V3.2` | 571 | 535 | 425 | 4.75 |
| All transitive events | 1,950 | 1,928 | | |

`V3.1` and `V3.2` fall from 75% of the transitive events to 62%. They still lead, for a reason that the verb-first draw does not touch: they are possible in nearly every scene (461 and 425 of 500), and the other verbs in fewer than half. Within a scene, every possible verb is now equally likely. The CAN features even out in the same way: the most frequent falls from 220 events to 158, and the least frequent rises from 9 to 17. Bringing `V3.1` and `V3.2` down further would take `scene.verb_weights`, or scenes whose participants are chosen with the rarer verbs in mind.

## Questions for Jon

### A. When does a class-level capacity take "can"?

The grammar has both `VP → V` and `VP → can V`. The examples in the specification use both for the same kind of proposition: "penguins swim", "red penguins swim", "owls eat mice", and "owls hunt mice" beside "all penguins can swim", "most red penguins can swim", and "things with wings and with feathers can fly". The logical form is the same either way.

Two cases are settled by the grammar itself. An instance-level capacity always takes `can`, because a bare verb after an instance reports an event ("the penguin can swim" against "the penguin swims"). A negative capacity always takes `can`, to carry `not`.

Options for a positive class-level capacity:

1. **A rate.** `grammar.class_can_rate`: the share of such sentences that take `can`. A rate of 1 gives "penguins can swim" always, and a rate of 0 gives "penguins swim" always.
2. **Always `can`.** Then `can` marks every sentence that is not an event.
3. **Never `can`**, except with `not`.
4. **By the predicate:** CAN features take `can`, and verbs do not.

**Recommendation and what is built:** option 1, with a default of 0.5. It covers options 2 and 3 as settings, and it matches the mix in the examples. The draw comes from the grammar stream, and both forms read back as the same logical form, so the setting never changes what a sentence says.

### B. What does a relative clause mean in a class-level sentence?

The specification says that a relative clause "expresses a true proposition of the same level as its sentence, about the same referent, and the logical form records it as a restriction". For an event-level or instance-level sentence that is clear: "the dog that chased the cat ran" reports two events, and each is true. For a class-level sentence, "a true proposition about the same referent" and "a restriction" pull apart. Take "owls that eat mice are big":

1. **A restriction.** The subject set is the owls that eat mice, and the predicate is grounded over that set, like "red owls are big". The quantifier is then chosen for the restricted set. This needs a rule for which owls "eat mice": an owl that can eat at least one mouse, or most mice.
2. **Pairs.** The propositional rendering `GEN(C_owl(X.1) AND C_mouse(X.2) AND V(X.1, X.2), SC.1.HIGH(X.1))` reads as a statement about pairs: for most pairs of an owl and a mouse where the owl eats the mouse, the owl is big. An owl then counts once for every mouse it eats.
3. **A second fact.** The clause states a second true class-level fact about the same category ("owls eat mice", and "owls are big"), and restricts nothing. The sentence is true when both facts are.

**Recommendation:** option 1, with "at least one": the subject set is the members that have the relation with at least one member of the other category. It is the ordinary restrictive reading, it extends the stage 2 truth tests directly, and vacuous sentences are already ruled out. A feature clause would work the same way ("penguins that can swim have fins").

**What is built:** nothing is drawn for class-level noun phrases yet. The grammar already realizes and reads back such a clause, so only the drawing and the truth test wait for the answer. Class-level sentences still have relative clauses where a restriction has negated IS literals (decision 27).

### C. One affix for each word

A word form takes one affix (`W.12.AF.1`). A verb can call for more: the past, the progressive, and the agreement marker.

**What is built:**

- A verb that carries a tense or aspect marker takes no agreement marker, as in English "chased". This holds whether the markers are affixes or words.
- A verb after `can` takes no agreement marker.
- With the past as the event tense, tense and aspect cannot both be affixes: the configuration is an error that asks for one of them to be a word. The default has tense as an affix and aspect as a word.

The other option is to let affixes stack ("chase-PAST-PROGRESSIVE"), which the word-form pipeline would then have to build in stage 7.

## Choices made by Claude

These are engineering choices made while building stage 4. Each one is the working design unless Jon changes it.

1. **Sentence plans.** What a sentence says is a sentence plan (`grammar.py`): a subject noun phrase and a predication, with the noun, the determiner or pronoun, the restriction, and the relative clause of every noun phrase. The planner makes the plan, and the grammar realizes it. `interpret(tree)` gives the plan back.
2. **What the tree cannot give.** `interpret` takes the tree, the lexicon, the referent of every noun phrase, and the event of every verb phrase. The words cannot say which penguin "the penguin" is, or which event a verb reports, so the sentence's record supplies them, in the order of the tree.
3. **Tree labels.** The subject and the object are `NP-SBJ` and `NP-OBJ`, so a tree reads the same in every word order. The object is inside the verb phrase when the verb is its neighbor, and a daughter of `S` in VSO and OSV.
4. **The order inside a noun phrase.** The noun phrase is built outward from the noun: adjectives, determiner, with-phrases, relative clause. The English default gives "the red penguin with fins that ...".
5. **The verb phrase.** The auxiliary, `not`, and the predicate word stay together. `has` and `is` count as auxiliaries, and their noun or adjective as the predicate word, so `word_order.auxiliary: after` gives "fins has" and "red is".
6. **The joined relative clause.** A verb phrase with the same auxiliary as the one before it does not say it again: "that are not red and not big", "that can swim and chase the cat". The negated IS literals come first.
7. **The object relative.** `that`, then the clause's subject and its verb in the order of `word_order.clause`: "that the owl chased" when the subject comes before the verb, "that chased the owl" when the verb comes first.
8. **Number.** Class-level nouns are plural and instances are singular. A part noun is never marked. A plural predicate noun takes no `a` ("penguins are birds"). With number off, "penguin is a bird".
9. **Tokens.** An inflected token is its lexeme label joined to the affix's gloss (`L.5-PLURAL`), and the formal rendering is `C1.3/L.5-PLURAL`. An inflection realized as a word is a token of its own, and a node beside the word it marks.
10. **The draws of the grammar.** Synonyms, the adjective order when it is not fixed, the aspect of an event, and the class-level `can` are drawn from the generator the caller gives, in one order whatever the word order. So a word-order setting never changes which words are chosen.
11. **The fixed adjective order** is one random ordering of every adjective concept (IS features, scalar poles, and patient projections), drawn once from `corpus:grammar:adjective_order`.
12. **Membership and the noun.** "The penguin is a penguin" is not a sentence. By default, the subject of "is a penguin" is named by the category above the leaf ("the bird is a penguin").
13. **Relative clauses for instances.** An instance-level clause states a capacity of the referent: a CAN feature, or a verb's relation with another referent of the document. An event-level clause reports another event of the same scene. A clause never repeats what the sentence already says. When the drawn kind (subject or object relative) has no true proposition, the other kind is used.
14. **Depth.** A relative clause on a noun phrase of the main clause has depth 1. `max_depth: 0` draws none. A restriction's negated IS literals are a relative clause of depth 1 whatever the limit, because they are part of what the sentence says.

## Notes for stage 5

- A pronoun cannot be the subject of an instance-level scalar pole, because the pole's comparison class is the category that the noun names.
- A noun phrase has one relative clause, so a noun phrase with negated IS literals can take a subject relative, which joins them, and never an object relative.
- The pronoun `it` is realized and read back, and the documents will bring it in.
