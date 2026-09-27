# Entity definitions

Draft, September 26, 2026. How entities are defined in the defined (scripted) world. The generative world will need its own system for "growing" living entities and its own rules for inanimate objects. That system is pinned for later. Jon's draft specification for the generative world's developing organisms is filed at `docs/specs/developmental-3d-artificial-animal-simulator-spec.md`.

## Four components

Every entity is assembled from up to four components:

| Component | What the component defines | Who has the component |
|---|---|---|
| **Body** | What the entity is physically: 3D model, size, mass, parts, physiology, needs. | Every entity: animals, humans, plants, and nonliving objects. |
| **Sensors** | What information reaches the entity: sight, hearing, smell, taste, touch, temperature, pressure, pain, internal state, and optionally propositions. | Animals and humans only. |
| **Actuators** | What the entity can do: walk, run, climb, fly, swim, slither, grasp, eat, drink, vocalize. | Animals and humans only. |
| **Nervous system** | The cognitive model that maps sensor input to actuator output, and learns. | Animals and humans only. |

Plants and nonliving entities have a body only.

The four components are independent modules. A species is a named combination of one body, a set of sensors, a set of actuators, and a nervous system. For example, the same human body could be run with different nervous systems (a symbolic planner, a neural network, a Bayesian model) without changing anything else. The same nervous system could, in principle, be placed in a human body or a giraffe body.

## Traits: fixed now, genetic later

Every numeric or categorical setting in every component is declared as a **trait**. In the defined world, each trait simply has a fixed value. Each trait also carries the fields needed to make the trait genetic later, so evolution can be switched on without redesigning the definitions:

| Trait field | Meaning | Used now? |
|---|---|---|
| `value` | The trait's value for this species. | Yes |
| `range` | The legal range of values. | Yes, for validation |
| `heritable` | Whether offspring inherit the trait from parents. | Later |
| `mutable` | Whether the trait can mutate. | Later |
| `variation` | How much the trait varies across individuals of the species (for example, a standard deviation). | Optional |
| `visible` | Whether other agents can perceive the trait (for example, fur color is visible and learning rate is not). | Yes |

This design follows Jon's earlier Dynamica-Python project, which declared each species' traits in a `trait_init_dict` with a size, a type, an initial mean and standard deviation, a mutable flag, and a visible flag. Dynamica's traits covered the body (size, teeth, fur, coloring), the nervous system (hidden-layer size, learning rates, action biases), and reinforcement set points (the target level of each drive). The same breadth applies here: sensor and actuator parameters can be traits too, so eyes, legs, and wings can eventually evolve.

## Body

The body defines what the entity is physically.

- **Model.** A reference to a stored 3D model, for example `human`, `giraffe`, or `elm_tree`. Models are created by us and kept in an asset library.
- **Phenotypic variation.** Traits such as size, fur color, or hair color. Which variations are possible depends on the asset library: either several model variants exist, or a generative process can modify one model (for example, scaling it or recoloring it).
- **Size and mass.** Mass matters for actuators (climbing, flying) and for being carried or eaten.
- **Parts.** A named list of body parts, such as head, torso, left arm, right leg, tail, wings, or fins. Sensors and actuators attach to parts. When touch, pain, or temperature are reported locally, the reports are per part.
- **Physiology.** Need levels and how the needs change over time (hunger, thirst, fatigue, temperature), temperature tolerance, insulation (for example, from fur), health, injuries, and age.
- **Material properties.** What the body provides if eaten, how strong the body is as a support (so a branch can break under a climber), and similar properties.

Plants add growth traits (growth rate, regrowth after being eaten, seasonal change). Nonliving objects (rocks, water sources, shelters) are bodies with no physiology.

### 3D models and variants

Realism is not the goal, and agents see small rendered images, so models are low-poly and stylized.

- **The simulation body and the visible body are separate.** The simulation uses simple collision shapes (one capsule per body part), mass, and the parts list. The 3D model is used only for rendering. The simulation never depends on the art, so the art can change without changing results.
- **Placeholders first.** The first milestone uses colored primitive shapes: for example, a tall capsule for a human, a green cylinder for a tree, and a blue plane for water. Real models replace the placeholders later.
- **Free, openly licensed libraries first.** Real models come first from libraries such as Quaternius and Kenney (low-poly characters, animals, and nature, some with animations) and Poly Haven (textures and nature assets). Each pack's license is checked before use, because the repository is public.
- **Plants, rocks, and terrain are generated in code**, for example with branching rules for trees and noise for terrain.
- **Custom models are made in Blender when needed**, with Blender's Python scripting for batch variants. AI text-to-3D tools are an option for prototypes.
- **One file format: glTF.** Bevy, three.js, and Blender all handle glTF natively.
- **Phenotypic variation comes from parameters, not separate files.** Size is scaling. Fur and hair color are color or texture swaps. Body shape uses blend shapes (morph targets). Optional parts, such as antlers or a mane, are switched on or off. Separate model files are needed only for different body plans. Parameter-based variation makes each phenotype a list of numbers, which is what genes can control later.
- **Animations are shared within a body plan**, for example walk, run, eat, sleep, climb, and swim for all quadrupeds. Without physics, animation is only cosmetic.
- **An asset manifest lives in the repository.** The manifest records each model's source, license, body plan, parts, and available variations. The large model files stay in Box. Curating and importing models is a well-bounded task for RAs.

## Sensors

Each sensor is a module that produces one named block of input. At each moment, all of an agent's sensor blocks together make up one big input vector. The engine also publishes a **sensor manifest** for each species, listing which block occupies which positions in the vector, so every model knows what each input means.

Sensors are rigidly defined for now. Every sensor parameter is a trait, so sensors can become genetic later.

| Sensor | If present, the agent receives | Example parameters |
|---|---|---|
| Eyes | A visual input array rendered from the agent's point of view. Rendered vision is included in the first milestone. | Field of view, resolution, range, color or grayscale. |
| Ears | Auditory input: sound energy per frequency band, per ear. See "Audition" below. | Number of ears, frequency bands, sensitivity. |
| Smell | Odor intensities by odor type, possibly with direction. | Range, which odor types. |
| Taste | Taste qualities of what is in the mouth. | Which taste qualities. |
| Touch | A touch vector: contact intensity and direction. Reported for the whole body or per body part (see "Touch, pain, and temperature" below). | Level of detail, which parts are sensitive. |
| Temperature | Temperature for the whole body, or local temperature at specific parts (is this water cold, is this food hot). | Level of detail, sensitivity. |
| Pressure | Pressure on body parts. | Sensitivity. |
| Pain | Damage signals: intensity, without direction. Reported for the whole body or per body part. | Level of detail, sensitivity. |
| Interoception | Internal need levels (hunger, thirst, fatigue, body temperature). | Exact or noisy readout. |
| Proprioception | Body posture and the agent's own last action. | — |
| Propositional sensor | A symbolic description of the agent's surroundings, as PDDL-compatible facts. | Range, level of detail. |

### Touch, pain, and temperature

How finely touch, pain, and temperature are reported is an option, set per species or per experiment. We start general:

- **Whole body.** One touch vector (intensity, plus direction relative to the body), one pain value, and one temperature value. The engine only needs to detect that the body collided with something.
- **Per body part.** The same readings for each named part: head, torso, each hand, each foot, and so on. Local readings allow distinctions such as "this water is cold on my hands" or "this food is hot in my mouth". The engine then needs to detect collisions per body part.
- **Finer surface patches.** A later option, if needed.

Pain works like touch, without the direction component. Temperature reports the temperature of whatever a body part is touching, and otherwise the air temperature.

### Audition

Jon's audio specification is filed at `docs/specs/3D_FDTD_Acoustic_Simulation_Implementation_Specification.docx`, as a suggestion rather than a requirement. That specification describes a full wave simulation of sound: pressure and velocity fields on a 3D grid, solved on a GPU. A full wave simulation captures reflection, diffraction, and room modes. The specification itself notes that the solver is meant for offline use, and that the cost grows roughly with the fourth power of grid resolution. A full wave simulation is therefore far too slow to run live for every agent in a training world.

Decided: the first version uses a simpler model:

- **Sound events.** Actions and entities emit sound events: footsteps, vocalizations, speech, running water, a predator's growl. Each event has a position, a start time, a loudness, and a spectrum (energy in a small number of frequency bands, for example 8 to 16 bands).
- **Propagation.** Loudness falls with distance. Arrival is delayed by distance divided by the speed of sound. Obstacles between source and listener muffle the sound, with high frequencies muffled more than low frequencies. Water and air can differ.
- **Ears.** Each ear reports energy per frequency band. Having two ears, one on each side of the head, gives direction cues: the nearer ear hears the sound louder and slightly earlier.
- **Speech.** Speech between agents travels as sound events. Whether a listener also receives the words as symbols is a separate option, like the propositional sensor.

The full wave solver in Jon's specification stays available as a later, higher-fidelity option. The most practical use is offline: precomputing how sound travels through a fixed piece of terrain, then applying the precomputed results during live simulation. The specification's own design already points that way, with precomputed impulse responses and a hybrid of wave and ray methods.

The **propositional sensor** is a deliberate shortcut. The propositional sensor feeds in information in a more symbolic format than any biological sense. The propositional sensor is still useful: symbolic models need it, and comparing agents with and without the propositional sensor measures how much a symbolic description helps. Experiments should always report whether the propositional sensor was switched on.

## Actuators

Each actuator is a module attached to one or more body parts. Actuators are rigidly coded, with simple rules instead of physics. Each actuator contributes actions to the agent's list of possible actions. Each action has an energy cost, a speed, and conditions for success.

**Impossible actions** are handled by an option. Either impossible actions are masked out, so the agent can only choose actions that are possible, or the agent may attempt any action and an impossible action simply fails. Allowing failed attempts, as Dynamica's notes proposed, lets agents learn what is possible.

| Actuator | Actions | Example rules |
|---|---|---|
| Legs | Walk, run, turn. | Speed depends on leg length and body mass. |
| Arms and hands | Grasp, carry, drop, climb, throw. | Climbing succeeds only if the thing climbed can support the climber's mass. Carrying is limited by strength relative to the load. |
| Wings | Fly, glide. | Flight is possible only if wing area is large enough relative to body mass. |
| Fins and tail | Swim. | Speed and energy cost depend on the medium. |
| Whole-body undulation | Slither, swim. | Used by snakes, eels, and similar bodies. |
| Mouth | Eat, drink, bite, carry in mouth. | What can be eaten depends on the body (teeth, digestion). |
| Vocal apparatus | Vocalize, speak. | Range and loudness. |

**Movement through media.** Each body has a speed and an energy cost for each medium: land, water, air, and trees. The table is what distinguishes a fish, a crocodile, a snake, and a giraffe without any real physics. A fish is fast in water and cannot leave the water. A crocodile moves well on land and very well in water. A snake slithers on land and can swim. This idea comes from Dynamica, where movement and resting costs were meant to vary by terrain and species, and where an animal could drown if its energy ran out in water.

## Nervous system

The nervous system is defined as broadly as possible. The nervous system is whatever cognitive model the agent uses: a hard-coded symbolic model, a planner, a neural network, a Bayesian model, a hybrid, or a scripted policy.

Every nervous system meets the same small interface:

- **Receive input.** The input vector from the agent's sensors (and, for models that want the input in structured form, the same input as named blocks).
- **Return output.** An action choice with its arguments, such as a direction, a speed, or a target.
- **Learn** (optional). Update itself from experience.
- **Report internals** (optional). Values, prediction errors, policy probabilities, and predictions, for logging and the viewer.
- **Save and load** (optional). For checkpoints at points in a lifetime.

The world does not hand out reward. If a nervous system learns from reward, the nervous system defines its own reward from the agent's internal state. Dynamica already worked this way: each animal's reinforcement came from the distance between its drive levels and genetically set target levels, so animals were not prewired to know that energy was good.

Nervous-system parameters (network size, learning rates, action biases, need set points) can also be traits, so they can evolve later.

## Examples

Entity types are built in code from parameters, so new types and variants can be created dynamically, without a file for each one. Files store only the default values, and the allowed ranges, of the parameters. An experiment can, for example, create 30 humans with masses drawn from the allowed range, or define a new type on the fly. The default-value files for a human, a giraffe, an elm tree, and a rock might look like this:

```yaml
species: human
body: {model: human, mass: 70, size: 1.0, parts: [head, torso, arms, hands, legs], insulation: 0.2}
sensors: [eyes, ears, smell, taste, touch, temperature, pain, interoception, proprioception]
actuators: [legs, arms_and_hands, mouth, vocal_apparatus]
nervous_system: {type: symbolic_planner, action_model: given}
```

```yaml
species: giraffe
body: {model: giraffe, mass: 800, size: 1.0, parts: [head, neck, torso, legs, tail], insulation: 0.4}
sensors: [eyes, ears, smell, taste, touch, temperature, pain, interoception]
actuators: [legs, mouth]
nervous_system: {type: scripted_forager}
```

```yaml
species: elm_tree
body: {model: elm_tree, size: 1.0, growth_rate: 0.01, support_strength: 900, seasonal: true}
```

```yaml
species: rock
body: {model: rock_small, mass: 5}
```

## What Dynamica-Python contributes

Jon's earlier project is at `https://github.com/jonwillits/Dynamica-Python` (Python, GPL-3.0, last updated June 2019). Ideas worth carrying over:

- The split of an animal into genome, phenotype, drive system, action system, and nervous system, which is close to the four components here.
- Species traits declared with mutable and visible flags, and visible traits feeding an appearance vector other animals can perceive.
- Genetically set reinforcement targets for each drive, which is homeostatic reinforcement learning with evolvable set points.
- A recurrent network that learns by prediction (of next sensory state, next drive state, and its own action) and by reinforcement, which is directly relevant to one of the project's research goals: understanding how prediction learning and reinforcement learning shore up each other's weaknesses.
- Ideas from Dynamica's notes: movement and resting costs that vary by terrain and species; drowning; letting agents attempt impossible actions (the action simply fails) instead of masking impossible actions, so agents learn what is possible; a possible "frustration" drive; and splitting energy into hunger and energy, with drives treated separately from traits.

Dynamica is licensed under GPL-3.0. Jon owns the code, so reusing Dynamica code in Semantic World under a different license is Jon's call.

## Decided

- **Impossible actions.** An option: masked out, or allowed and simply fail.
- **Touch, pain, and temperature detail.** An option, starting general: whole-body readings first, per-body-part readings as a setting.
- **Rendered vision.** Included in the first milestone.
- **How types are defined.** Entity types are built in code from parameters and can be created dynamically. Files store only default values and allowed ranges.
- **Audition.** The first version uses the simple sound-event model. The full wave solver is kept as a later, higher-fidelity option, most likely for offline precomputation.
- **Default-value file syntax.** YAML (decided 2026-09-27; see `CONTRACTS.md`).
- **3D models.** Low-poly and stylized; placeholders first; free, openly licensed libraries first; glTF; variation through parameters. See "3D models and variants" above.
