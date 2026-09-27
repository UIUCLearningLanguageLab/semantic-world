# Developmental 3D Artificial-Animal Simulator

Implementation-oriented specification for an evolvable simulator of developing, sensing, controlled, moving 3D organisms.

**Status:** Draft for implementation  
**Specification version:** 0.1  
**Target:** A deterministic research simulator and evolutionary testbed  
**Primary phenotype representation:** Connected 3D cell-particles with local mechanical bonds  
**Normative language:** “MUST”, “SHOULD”, and “MAY” indicate required, recommended, and optional behavior.

---

## 1. Executive summary

The simulator evolves genomes that do not directly describe body parts or final geometry. A genome contains local regulatory rules. Starting from one or a small number of seed cells, those rules control division, growth, death, differentiation, adhesion, morphogen production, and neural outgrowth. Development produces both:

1. a mechanical body graph, whose nodes are cells and whose edges are local bonds; and
2. a neural signaling graph, whose nodes are neural or receptor/effector endpoints and whose edges can span beyond mechanical neighbors.

Cells occupy a continuous property space. “Bone”, “muscle”, “skin”, and “neuron” are required tissue presets or attractors within that space, not unrelated simulation object classes. This permits intermediate materials such as tendon-like tissue without adding a special primitive.

Locomotion emerges from physics. Muscle activation shortens selected bonds’ preferred lengths, creating forces against stiffer tissues. Joints are not named or hard-coded; they emerge where rigid regions are connected by softer or differently connected material. Skin exposes local environmental measurements to the neural graph, and tissue stretch supplies proprioception. A global organism energy pool initially abstracts circulation and resource transport.

The first implementation SHOULD favor clarity, reproducibility, and evolvability over biological fidelity. It MUST support exact replay from a genome, configuration, simulator version, and seed.

---

## 2. Design principles

1. **Developmental, not constructive:** the genome specifies local rules, not limbs, joints, meshes, or target coordinates.
2. **Local information:** regulatory conditions observe only cell state, local neighbors, morphogens, local mechanics, developmental time, lineage, and local environment.
3. **Two graphs:** mechanical adjacency and neural connectivity are distinct data structures with distinct creation and update rules.
4. **Properties over taxonomies:** tissue labels provide defaults and diagnostics; physics uses material and functional properties.
5. **Emergent articulation:** no `Joint`, `Limb`, `Eye`, or `Gait` phenotype primitive is permitted in the MVP.
6. **Small mutations should usually have bounded effects:** numeric encodings, developmental rate limits, and repair rules SHOULD make nearby genomes tend to produce nearby phenotypes.
7. **Determinism is a feature:** reproducibility MUST survive normal multithreaded execution on a supported platform.
8. **Invalid development is data:** malformed or nonviable organisms receive explicit failure reasons and bounded fitness, rather than crashing or hanging the simulation.
9. **Inspectable causality:** developers MUST be able to answer which rule caused a cell action, where a morphogen came from, and what activated a muscle.
10. **Layered fidelity:** simple models must have clear replacement points for later continuum mechanics, nutrient transport, spiking neurons, and richer sensors.

---

## 3. Goals and non-goals

### 3.1 Goals

The system MUST:

- develop a 3D organism from a seed configuration and a genome;
- support bounded cell growth, division, differentiation, apoptosis, bond formation, and bond removal;
- support at least bone-, muscle-, skin-, and neuron-like tissue behavior;
- diffuse several morphogen channels over local tissue topology;
- simulate tissue-specific particle-and-bond mechanics, collisions, gravity, and contact with an environment;
- allow muscle cells to alter selected bonds’ preferred lengths under neural control;
- construct and simulate a neural graph independently of mechanical adjacency;
- expose skin/environment sensing and tissue-stretch proprioception to the neural controller;
- account for maintenance, development, neural activity, and actuation using a global energy pool;
- evaluate organisms in reproducible tasks and expose mutation/evolution interfaces;
- record sufficient information for replay, inspection, and regression testing;
- run many independent organism evaluations in parallel.

The system SHOULD:

- allow development to form disconnected regions, holes, branches, cavities, compliant joints, and repeated structures;
- allow ongoing adaptation or growth during evaluation, while supporting a simpler develop-then-evaluate mode;
- make most genome mutations syntactically valid by construction;
- provide headless batch execution and an interactive visual debugger using the same simulation core.

### 3.2 Non-goals for the MVP

The MVP does not attempt:

- molecular biology, gene transcription kinetics, or biologically accurate cell division;
- blood vessels, oxygen transport, digestion, immune systems, healing, or reproduction;
- accurate finite-element tissue mechanics or incompressible fluid dynamics;
- anatomical primitives such as bones, limbs, hinges, eyes, spinal cords, or central pattern generators;
- spiking neuron models, neurotransmitter chemistry, or detailed axon morphology;
- photorealistic rendering;
- real-time performance for a single very large organism;
- guaranteed evolution of sophisticated locomotion without curriculum and fitness shaping;
- exact cross-architecture floating-point identity in the first release. Exact replay is required only within a declared deterministic platform profile; cross-platform results MUST remain statistically and structurally comparable.

---

## 4. Terminology and simulation units

| Term | Meaning |
|---|---|
| Cell | A simulated 3D particle with developmental, material, signaling, and optional neural state. |
| Mechanical graph, `G_mech` | Undirected graph of cells and load-bearing/local adhesion bonds. |
| Neural graph, `G_neural` | Directed weighted graph carrying control signals; it is not derived solely from mechanical adjacency. |
| Genome | Evolvable regulatory program and heritable parameter set. |
| Rule | A local condition plus one or more bounded developmental actions. |
| Tissue preset | Named initialization region in continuous property space. |
| Morphogen | A scalar local signal that is produced, diffused, decayed, and sensed during development. |
| Development tick | One regulatory and morphogen update interval. |
| Control tick | One neural update and sensor/actuator interval. |
| Physics substep | One stable numerical integration step. |
| Phenotype | The developed cells, graphs, state, and resulting behavior. |

The engine MUST use one documented unit system. Recommended defaults are:

- length: meter;
- mass: kilogram;
- time: second;
- force: newton;
- energy: joule-like simulation unit;
- morphogen concentrations and neural activations: normalized dimensionless values, normally `[0, 1]` or `[-1, 1]` as declared.

Default parameter values MAY use artificial scales, but all stored quantities MUST be interpreted consistently.

---

## 5. System architecture

```text
Genome + Seed + Run Configuration + RNG Seed
                    |
                    v
           Development Engine
       / regulatory rules / morphogens
      / division / death / differentiation
                    |
                    v
            Developed Phenotype
       +------------+-------------+
       |                          |
       v                          v
 Mechanical Graph           Neural Graph
 bonds / materials       sensors / neurons / motors
       |                          |
       +------------+-------------+
                    v
             Runtime Scheduler
     environment -> sensing -> control
       -> actuation -> physics -> energy
                    |
                    v
       Metrics / Fitness / Replay / Debugging
```

### 5.1 Required modules

| Module | Responsibility |
|---|---|
| `genome` | Schema, validation, decoding, rule execution, mutation, crossover. |
| `development` | Cell lifecycle, division, differentiation, apoptosis, lineage, morphogen sources, surface classification. |
| `morphogens` | Local diffusion, decay, production, and optional environmental fields. |
| `topology` | Mechanical bonds, neural edges, adjacency, spatial indexing, deterministic topology edits. |
| `mechanics` | Forces, constraints, collisions, integration, damage, contact. |
| `tissues` | Material/property presets, interpolation, receptor and effector profiles. |
| `neural` | Neural growth, synapse formation, continuous activation dynamics. |
| `sensors` | Skin sensing, proprioception, optional internal-state sensors. |
| `energy` | Global energy budget, cost ledger, starvation policy. |
| `environment` | Terrain, fields, resources, contacts, task API. |
| `scheduler` | Multirate timestep ordering and lifecycle phases. |
| `evolution` | Evaluation contract, mutation hooks, population orchestration, lineage metadata. |
| `serialization` | Genome, checkpoint, phenotype summary, replay and version migration. |
| `observability` | Metrics, event traces, deterministic hashes, visualization data. |

The simulation core MUST run headlessly. Visualization MUST consume public snapshots or event streams and MUST NOT alter simulation state.

### 5.2 Recommended lifecycle modes

- **Develop then evaluate (MVP default):** develop for a fixed number of ticks or until quiescence, freeze topology-changing developmental actions, settle mechanics, then evaluate behavior.
- **Continuous development (later):** development, healing, growth, and behavior remain active together. This mode requires explicit energy competition and stricter stability controls.

---

## 6. Core entities and data models

The following language-neutral structures define required logical fields. Implementations MAY use structure-of-arrays storage, packed indexes, handles, or ECS components internally.

### 6.1 Identifiers

```text
OrganismId  : stable 128-bit or 64-bit identifier
CellId      : stable monotonically allocated integer within an organism
BondId      : stable monotonically allocated integer within an organism
NeuralEdgeId: stable monotonically allocated integer within an organism
RuleId      : stable identifier within a genome
LineageId   : derived from parent lineage plus deterministic division ordinal
```

IDs MUST NOT be reused within a run. Dense array indexes MUST NOT be exposed as stable IDs.

### 6.2 Cell

```text
Cell {
  id: CellId
  parent_id: CellId?
  lineage_id: LineageId
  birth_tick: uint64
  age: float
  alive: bool

  position: Vec3
  velocity: Vec3
  orientation: Quaternion
  polarity: Vec3
  radius: float
  mass: float

  tissue_label: TissueLabel       // diagnostic/defaults, not physics dispatch
  tissue_mix: float[T]            // optional normalized continuous mixture
  material: MaterialProperties
  functions: FunctionalProperties

  morphogen: float[M]
  morphogen_next: float[M]
  regulator_state: float[R]
  receptor_expression: float[S]
  emitter_rates: float[M]

  cycle_phase: float
  division_cooldown: float
  growth_rate: float
  health: float
  damage: float

  neural: NeuralNodeState?        // present when neural function is expressed
  energy_demand_accumulator: float
}
```

Required invariants:

- position, velocity, radius, mass, and every stored scalar MUST remain finite;
- radius and mass MUST remain within configured positive bounds;
- morphogen and expression values MUST be clamped to declared ranges;
- a dead cell MUST participate in no future regulatory, mechanical, or neural update after its deterministic cleanup boundary;
- differentiation MUST modify properties through a bounded transition, not silently replace cell identity.

### 6.3 Material and functional properties

```text
MaterialProperties {
  density: float
  young_like_stiffness: float
  damping: float
  adhesion: float
  tensile_strength: float
  compressive_strength: float
  bending_resistance: float
  friction_static: float
  friction_dynamic: float
  restitution: float
  collision_group: uint32
}

FunctionalProperties {
  contractility: float
  max_contraction_fraction: float
  contraction_rate: float
  neural_conductivity: float
  axon_growth_rate: float
  sensory_gain: float[S]
  morphogen_diffusivity_scale: float[M]
  maintenance_cost: float
  growth_cost_scale: float
}
```

All evolvable physical properties MUST have safe lower and upper bounds. Mutations SHOULD operate in normalized or log space where orders of magnitude are meaningful.

### 6.4 Mechanical bond

```text
MechanicalBond {
  id: BondId
  a: CellId
  b: CellId
  created_tick: uint64
  base_rest_length: float
  current_rest_length: float
  stiffness: float
  damping: float
  tensile_limit: float
  compressive_limit: float
  bending_weight: float
  muscle_influence_a: float
  muscle_influence_b: float
  damage: float
  enabled: bool
}
```

There MUST be at most one primary mechanical bond per unordered cell pair in the MVP. Bond parameters SHOULD be derived symmetrically from both endpoint cells and genetically controlled adhesion state.

### 6.5 Neural node and edge

```text
NeuralNodeState {
  activation: float
  next_activation: float
  bias: float
  leak: float
  gain: float
  node_role: {INTERNEURON, SENSOR_RELAY, MOTOR_RELAY, MIXED}
  axon_tips: AxonTip[]
}

NeuralEdge {
  id: NeuralEdgeId
  source: CellId
  target: CellId
  weight: float
  delay_ticks: uint16
  plasticity: float             // zero in MVP unless explicitly enabled
  enabled: bool
}

AxonTip {
  position: Vec3
  direction: Vec3
  path_length: float
  state: {GROWING, CONNECTED, STALLED, RETRACTED}
  guidance_profile_id: uint16
}
```

Neural edges MUST be directed and MUST remain valid independently of mechanical bond changes. If an endpoint dies, the edge MUST be removed at the neural cleanup boundary.

### 6.6 Organism state

```text
Organism {
  id: OrganismId
  genome_hash: Hash
  generation: uint64
  developmental_tick: uint64
  control_tick: uint64
  physics_tick: uint64
  cells: CellStore
  mechanical_bonds: BondStore
  neural_edges: NeuralEdgeStore
  energy: EnergyState
  event_queue: DeterministicEventQueue
  rng_streams: RngStreams
  status: OrganismStatus
  failure_reason: FailureReason?
  metrics: OrganismMetrics
}
```

### 6.7 Deferred topology commands

Topology MUST NOT be changed while iterating the live graph. Systems emit commands into per-system buffers:

```text
TopologyCommand =
  DivideCell(parent, parameters, cause)
  KillCell(cell, cause)
  AddBond(a, b, parameters, cause)
  RemoveBond(bond, cause)
  AddNeuralEdge(source, target, parameters, cause)
  RemoveNeuralEdge(edge, cause)
```

At a documented synchronization boundary, commands are stable-sorted by command class, target IDs, rule priority, and `RuleId`; conflicts are resolved deterministically and then committed.

---

## 7. Tissue model

### 7.1 Continuous property space

Physics and control code MUST read cell properties rather than switch exclusively on `tissue_label`. Labels may select defaults, color visualizations, seed mutation priors, and support metrics.

Required presets:

| Preset | Mechanical profile | Functional profile | Typical connectivity | Primary role |
|---|---|---|---|---|
| Bone | very stiff, high compression strength, low strain tolerance | no contraction, low signaling | dense local bonds | load-bearing structure |
| Muscle | compliant-to-medium stiffness, high damping | active contraction, moderate energy cost | aligned load-bearing bonds | actuator |
| Skin | compliant, damage-sensitive, frictional | environmental receptors | surface sheet/network | boundary and sensing |
| Neuron | mechanically weak or neutral | neural activation and axon growth | sparse mechanical, nonlocal neural | control and signal routing |

The implementation SHOULD also define a generic undifferentiated preset. Tendon-like material MUST be representable using properties—high tensile strength, moderate flexibility, no active contraction—without requiring a dedicated engine class.

### 7.2 Differentiation

Rules target a tissue attractor or property delta. For target preset `P`, differentiation over rate `r` is:

```text
properties_next = lerp(properties_current, properties(P), clamp(r * dt_dev, 0, 1))
```

Abrupt differentiation MAY be allowed for tests but SHOULD not be the evolutionary default. Conflicting differentiation actions combine by normalized weighted blending after priority filtering.

### 7.3 Surface status

A cell is provisionally surface-exposed when one or more of the following is true:

- its neighbor count is below a configured local packing threshold;
- a sampled solid angle around it contains sufficient unoccupied space;
- its distance-to-exterior estimate is below a threshold.

The MVP SHOULD use neighbor count plus solid-angle sampling from the spatial index. Surface status MUST be a sensed condition, not a permanent type. It MAY have hysteresis to prevent flicker.

### 7.4 Damage and bond failure

Bond strain beyond configured tensile or compressive thresholds accumulates damage. When damage exceeds one, the bond is queued for removal. Cell damage may be based on contact impulse, excessive strain, or unsupported exposure. Damage MAY be disabled during early development but MUST be available during evaluation.

---

## 8. Genome and regulatory encoding

### 8.1 Requirements

The encoding MUST:

- specify local conditions and actions, not final body-part geometry;
- be serializable and versioned;
- produce syntactically valid offspring after supported mutation operators;
- bound the number and magnitude of actions a rule can cause per tick;
- expose neutral and incremental mutations;
- allow repeated motifs and reused regulatory subprograms;
- record stable `RuleId` values for causal tracing.

The initial genome SHOULD contain approximately 20–50 rules and 3–5 morphogen channels, while allowing configured limits beyond those values.

### 8.2 Genome structure

```text
Genome {
  schema_version: uint32
  constants: BoundedScalar[]
  tissue_presets: EvolvableTissueDelta[]
  regulatory_nodes: RegulatoryNode[]
  rules: DevelopmentRule[]
  neural_growth_programs: NeuralGrowthProgram[]
  receptor_profiles: ReceptorProfile[]
  mutation_parameters: MutationProfile
}
```

### 8.3 Regulatory inputs

Each cell evaluates a fixed, versioned input vector. Candidate channels include:

- normalized cell age and global developmental time;
- lineage depth and bounded lineage markers;
- current tissue mixture and regulatory state;
- morphogen concentrations and local gradients;
- neighbor count and neighbor-type summaries;
- surface exposure;
- pressure, mean bond strain, maximum strain, and local curvature proxy;
- energy availability and local damage;
- cell polarity and gravity-relative orientation;
- optional local environment channels during continuous development.

Global Cartesian coordinates MUST NOT be exposed to the genome in the default mode. If enabled for experiments, they MUST be declared as a separate treatment because they weaken translation invariance and developmental generality.

### 8.4 Regulatory nodes

Recommended MVP implementation:

```text
RegulatoryNode {
  id: uint16
  weighted_inputs: (InputChannel, weight)[]
  bias: float
  activation: {SIGMOID, TANH, CLAMPED_LINEAR}
  decay: float
}
```

Node state is updated synchronously from the previous developmental tick. A sparse small gene-regulatory network permits compositional logic while keeping mutations continuous.

### 8.5 Development rule

```text
DevelopmentRule {
  id: RuleId
  enabled: bool
  priority: int16
  condition: ConditionExpr
  refractory_ticks: uint16
  probability: float
  max_firings_per_cell: uint16?
  actions: Action[]
}

ConditionExpr = bounded expression tree over:
  comparisons(input_or_regulator, threshold)
  AND / OR / NOT
  optional smooth gate in [0, 1]

Action =
  Divide(direction_rule, size_ratio, polarity_rule)
  Grow(rate_delta)
  StopGrowth
  Apoptosis
  Differentiate(target_preset_or_mix, rate)
  EmitMorphogen(channel, rate)
  ChangeReceptor(channel, delta)
  ChangeAdhesion(delta)
  RotatePolarity(axis_rule, bounded_angle)
  CreateLocalBond(selector, parameter_delta)
  RemoveLocalBond(selector)
  StartAxon(guidance_program)
```

Conditions SHOULD use smooth gates when practical so small threshold mutations do not always cause discontinuous organism-wide changes. Boolean conditions remain permitted for clear developmental switches.

### 8.6 Rule evaluation and conflicts

For every development tick:

1. Read inputs from an immutable snapshot.
2. Update regulatory nodes into a second buffer.
3. Evaluate enabled rules in stable `RuleId` order.
4. Convert conditions to activation values in `[0, 1]`.
5. Apply refractory, firing-count, probability, energy, and rate-limit gates.
6. Emit actions with `RuleId`, priority, activation, and target cell.
7. Resolve actions by category and commit at the topology boundary.

Conflict policy:

- `Apoptosis` overrides division and growth for the same cell.
- At most one division action per cell per development tick is accepted.
- Multiple morphogen emissions add and then clamp.
- Compatible numeric property changes add in normalized space and clamp.
- Differentiation targets blend by activation after discarding lower-priority conflicting actions.
- `StopGrowth` overrides positive growth actions at equal or lower priority.
- Bond creation/removal conflicts resolve by priority, then lower `RuleId`.

Probabilistic gates MUST draw from a counter-based PRNG keyed by `(run_seed, organism_id, cell_id, rule_id, development_tick, draw_kind)`. This prevents iteration order from changing outcomes.

### 8.7 Cell division

An accepted division MUST:

1. verify cell-count, energy, size, cooldown, and local-overlap limits;
2. choose an axis from cell polarity, morphogen gradient, principal local stress, random unit vector, or a bounded blend;
3. place two cell centers along that axis with configurable separation;
4. conserve parent mass within tolerance, subject to explicitly charged growth mass;
5. partition internal state deterministically, with optional bounded noise;
6. preserve and mutate polarity as specified;
7. create or inherit local bonds using a declared policy;
8. allocate child IDs deterministically;
9. record a lineage and causality event.

Failed division attempts MUST be reported as bounded events, not retried indefinitely in the same tick.

### 8.8 Developmental robustness and evolvability safeguards

- Clamp every action and property to safe limits.
- Limit cells, bonds, neural edges, axon tips, rule firings, and emitted signal per organism.
- Make rate changes and numeric perturbations more common than structural rule mutations.
- Support rule duplication followed by divergence.
- Allow inactive rules and regulatory nodes to create neutral variation.
- Use developmental checkpoints and explicit viability metrics instead of only final task fitness.
- Include hysteresis/cooldowns for differentiation, division, death, and surface classification.
- Penalize explosive cell growth, isolated-cell debris, extreme material values, and unrecoverable self-intersection.
- Preserve modular `RuleId` and module tags through mutation for lineage analysis.
- Provide mutation-scale annealing and per-field mutation masks.

---

## 9. Developmental loop and morphogens

### 9.1 Development tick

```text
function development_tick(organism, environment):
    snapshot cell and graph state
    compute local neighborhoods and surface status
    compute mechanics-derived developmental inputs
    update morphogen production, diffusion, and decay
    compute morphogen gradients
    update regulatory network synchronously
    evaluate genome rules into action buffers
    reserve energy for accepted non-topology actions
    resolve and commit division/death/bond/topology commands
    update differentiation, growth, polarity, receptors, and emitters
    advance neural growth tips and resolve new neural edges
    rebuild dirty adjacency/spatial-index regions
    validate invariants and update development metrics
```

Development MUST terminate when any configured condition is met:

- maximum development ticks;
- maximum cells or other resource limit;
- quiescence for `N` ticks, defined as no topology changes and property change below epsilon;
- depleted energy with no legal recovery path;
- invalid-state or numerical-failure policy;
- optional genome-controlled maturation signal above threshold for `N` ticks.

### 9.2 Graph-based morphogen diffusion

The MVP SHOULD diffuse morphogens over the mechanical neighbor graph rather than over a 3D voxel grid. For cell `i`, channel `m`:

```text
c_i,m(t+dt) = clamp(
    c_i,m(t)
    + dt * [D_i,m * sum_j w_ij * (c_j,m - c_i,m)
            - lambda_m * c_i,m
            + source_i,m],
    0, c_max_m)
```

Where:

- `j` ranges over mechanically adjacent cells;
- `w_ij` is a normalized geometry/conductance weight;
- `D_i,m` is the base diffusion coefficient scaled by local tissue properties;
- `lambda_m` is decay;
- `source_i,m` is genetically regulated production.

Diffusion MUST use double buffering. The explicit update MUST obey a configured stability bound; otherwise use multiple substeps or a stable implicit/normalized scheme. Concentration MUST not depend on adjacency iteration order.

### 9.3 Morphogen gradients

A local gradient estimate SHOULD use weighted least squares over neighbor offsets. If the neighborhood is degenerate, return a zero gradient plus a diagnostic flag. Gradient directions used for division or axon guidance MUST be normalized safely.

### 9.4 Development and mechanics coupling

During development, the physics engine SHOULD run enough substeps between development ticks to relax severe overlaps without requiring full equilibrium. A configurable “developmental damping” MAY reduce violent motion. Mechanics-derived regulatory inputs are sampled only at the beginning of a development tick.

---

## 10. Physics and mechanics

### 10.1 Representation

Cells are spherical or ellipsoidal particles for collision and mass. Mechanical bonds are spring-damper constraints. The MVP MAY render cells as spheres even if orientation and polarity are stored.

### 10.2 Bond force

For bond `i-j`, displacement `x = p_j - p_i`, distance `d = |x|`, direction `n = x / max(d, epsilon)`, relative axial speed `v_rel = dot(v_j - v_i, n)`:

```text
F_ij = [k_ij * (d - L_ij) + c_ij * v_rel] * n
```

Equal and opposite forces MUST be accumulated for the two endpoints. Implementations MAY use a position-based dynamics or XPBD constraint form instead of explicit forces if it preserves the specified rest-length, stiffness/compliance, damping, and energy-accounting semantics.

### 10.3 Muscle actuation

For every muscle-influenced bond:

```text
a = clamp(weighted_motor_activation, 0, 1)
target_fraction = clamp(alpha * a, 0, max_contraction_fraction)
target_rest_length = base_rest_length * (1 - target_fraction)
current_rest_length = rate_limit(
    current_rest_length,
    target_rest_length,
    contraction_rate * dt_control)
```

`alpha` comes from endpoint contractility and bond influence. Relaxation toward base rest length MUST also be rate-limited. The energy system may reduce realized contraction when energy is insufficient. Muscles MUST not directly apply desired world-space motion or joint torque.

Antagonistic actuation can emerge from separately innervated muscle regions on opposing sides of a compliant connection.

### 10.4 Emergent joints

No joint object is generated from the genome. Articulation arises from:

- stiff, densely bonded cell regions;
- low-stiffness or sparse connector regions;
- directional muscle bonds;
- differences in bending resistance and adhesion;
- local topology and contact geometry.

The engine MAY provide non-genetic numerical constraints to stabilize very stiff bonds, but those constraints MUST be derived from material properties and connectivity and MUST NOT label anatomical joints.

### 10.5 Bending and volume preservation

The MVP SHOULD support at least one of:

- angle/bending constraints over local bond triplets; or
- shape-matching clusters inferred from highly stiff local neighborhoods.

Without this, a high-stiffness spring network may shear unnaturally. Optional local volume constraints MAY improve tissue behavior but must be topology-derived.

### 10.6 Collision and contact

Required collision pairs:

- cell versus static environment;
- cell versus resource/task objects;
- nonbonded cell versus cell, at least within the same organism.

Directly bonded neighbors MAY use softened or disabled collision response to prevent jitter. Collision broadphase SHOULD use a uniform spatial hash or BVH. Contact must expose normal impulse, penetration, and material/environment tags to sensors and damage logic.

### 10.7 Integrator and stability

Recommended MVP choices are semi-implicit Euler with substepping, XPBD, or an existing deterministic constraint solver. The engine MUST:

- cap or report excessive velocity and force;
- detect NaN/infinite state immediately;
- expose solver iteration count and residual metrics;
- use fixed timesteps in deterministic evaluation;
- provide a settling phase after development and before scoring locomotion.

### 10.8 Disconnected components

The organism may develop multiple mechanical components. The component containing the original seed is the primary body by default. Detached living components MAY remain simulated and consume energy. Evaluation policy MUST state whether their movement contributes to fitness. Debris MUST never be silently deleted if deletion would alter energy or score.

---

## 11. Neural growth and control

### 11.1 Separate graph requirement

The neural graph MUST be stored and updated separately from the mechanical graph. Mechanical neighbors do not automatically become neural neighbors, and neural edges may connect nonadjacent cells.

### 11.2 Neural differentiation

A cell expressing sufficient neural function receives `NeuralNodeState`. Losing neural expression MAY cause gradual retraction or immediate node removal depending on configuration; the selected behavior MUST be deterministic and recorded.

### 11.3 Axon growth

Each neural growth program defines a bounded direction field:

```text
direction = normalize(
    w_polarity * cell_polarity
  + sum_m w_attract[m] * grad(m)
  - sum_m w_repel[m] * grad(m)
  + w_target * local_target_signal
  + w_persistence * previous_direction
  + bounded_noise)
```

At each neural growth update, a tip advances by a bounded step. It forms a synapse when it enters the capture radius of a compatible neural, sensor-relay, or motor-relay target and passes genetic compatibility gates. Tips stop at maximum path length, energy budget, age, or stall count.

The MVP MAY approximate the axon as a path used only for growth and visualization; ongoing signal transmission can use the final graph edge. Delay SHOULD be derived from path length and conductivity.

### 11.4 Continuous neural dynamics

At control tick `t`:

```text
u_i(t) = bias_i
       + sensor_input_i(t)
       + sum_(j -> i) weight_ji * delayed_activation_j(t)

candidate_i = activation_function(gain_i * u_i(t))
a_i(t+1) = clamp((1 - leak_i) * a_i(t) + leak_i * candidate_i,
                 activation_min, activation_max)
```

Updates MUST be synchronous/double-buffered. A bounded ring buffer supports delays. Supported activation functions SHOULD include `tanh` and sigmoid. Spiking is explicitly deferred.

### 11.5 Sensor and motor coupling

- Receptor expression maps local sensor channels into a cell’s neural input.
- Neural output reaches muscle through direct local motor-relay expression or explicit neural edges ending on muscle cells.
- A muscle’s activation is the bounded weighted sum of its motor inputs.
- Sensor and motor gains MUST be evolvable within safe bounds.
- The engine MUST record source contributions for selected neural nodes during debug traces.

### 11.6 Neural failure handling

Cycles are legal because updates are synchronous. Unconnected neurons are legal but consume maintenance energy. Activations must remain finite and clamped. Edge count and fan-in/fan-out must be bounded.

---

## 12. Sensors

### 12.1 Skin receptors

Skin-expressed receptors MAY sample the following normalized local channels:

- contact pressure/normal impulse;
- tangential contact/slip;
- temperature;
- light intensity and optionally local light direction;
- environmental chemical or resource concentration;
- damage/pain;
- surface exposure.

The MVP MUST implement contact pressure and at least one distal/environmental scalar field such as light or resource concentration. Receptors sample only at the cell’s location and orientation; there is no global object list or target position input.

### 12.2 Proprioception

For cell `i`, bond-strain channels SHOULD include:

```text
strain_ij = (distance_ij - current_rest_length_ij)
            / max(current_rest_length_ij, epsilon)
```

Cells can sense mean strain, maximum tension, maximum compression, and strain-rate summaries over incident bonds. Muscle cells MAY additionally sense realized contraction and activation. These signals enter the neural graph through expressed proprioceptive receptors.

### 12.3 Internal sensors

Optional local channels include energy fraction, damage, age, morphogen concentration, and acceleration. The exact available set MUST be versioned as part of the experimental configuration.

### 12.4 Sensor normalization

Every channel MUST document units, clipping range, normalization, noise model, and missing-value behavior. Sensor noise MUST use keyed deterministic randomness and MUST be disabled by default in unit tests.

---

## 13. Energy and metabolism abstraction

### 13.1 Global pool

Each organism has one energy state:

```text
EnergyState {
  current: float
  capacity: float
  income_accumulator: float
  maintenance_spent: float
  development_spent: float
  actuation_spent: float
  neural_spent: float
  damage_spent: float
}
```

For scheduler interval `dt`:

```text
delta_E = energy_absorbed
        - cell_maintenance_cost
        - growth_and_division_cost
        - muscle_work_cost
        - neural_activity_cost
        - optional_repair_cost
```

This abstraction intentionally treats circulation and resource distribution as instantaneous within one organism.

### 13.2 Cost policy

- Maintenance scales with living cell properties and `dt`.
- Growth cost scales with created mass/volume.
- Division has a fixed organizational cost plus any growth cost.
- Positive mechanical work and/or activation drives muscle cost; the exact formula MUST be documented.
- Neural cost scales with node count, edge transmissions, and optionally activation magnitude.
- Morphogen production has a configurable cost.

Costs MUST be accumulated from accepted actions, not attempted actions, except a small configurable failed-attempt cost if desired.

### 13.3 Energy shortage

The default priority order is:

1. baseline maintenance;
2. neural sensing/control;
3. muscle actuation;
4. development and growth;
5. repair.

Within a class, demands are scaled proportionally if energy is insufficient. Starvation first suppresses growth and contraction, then reduces neural activity, then applies health loss. Starvation behavior MUST not depend on cell iteration order.

### 13.4 Resource acquisition

The environment exposes explicit resource contacts or scalar fields. Absorption is based on local receptor/absorber expression and contact, then credited to the global pool. The MVP MAY initialize energy without implementing feeding if the first task is fixed-duration locomotion.

### 13.5 Future replacement point

The `EnergyProvider` interface MUST permit later replacement by cell-local nutrient/oxygen fields and transport networks without changing the genome rule API wholesale.

---

## 14. Environment interface

```text
Environment {
  initialize(config, seed)
  query_static_collision(shape, transform) -> contacts
  sample_field(channel, position, orientation?) -> value
  consume_resource(resource_id, amount) -> accepted_amount
  apply_interaction(command) -> result
  step(dt)
  task_observation() -> evaluator-only data
  serialize_state()
}
```

The organism MUST NOT receive `task_observation`; it is reserved for scoring and diagnostics. Cells receive only declared local sensor channels.

### 14.1 MVP environment

The baseline environment SHOULD include:

- gravity;
- an infinite or large planar ground with friction;
- optional slopes or simple static obstacles;
- a directional or point scalar stimulus;
- optional collectible/contact resource objects;
- configured spawn transform and boundary policy.

### 14.2 Task evaluator

```text
TaskEvaluator {
  begin_episode(initial_snapshot)
  observe_step(read_only_snapshot)
  end_episode(final_snapshot) -> FitnessVector + Diagnostics
}
```

Example locomotion metrics:

- primary-body center-of-mass displacement along a target axis;
- displacement per unit energy;
- uprightness or height retention, if task-appropriate;
- damage and cell-loss penalties;
- developmental viability and controller connectivity;
- penalty for crossing cell/bond/resource limits.

Fitness MUST be computed outside the organism controller. Multiobjective vectors SHOULD be retained even when a scalar selection score is used.

---

## 15. Simulation timestep ordering

### 15.1 Fixed multirate schedule

Recommended starting values, subject to stability tests:

- `dt_physics`: 0.001–0.005 s;
- `dt_control`: 0.01–0.05 s;
- `dt_development`: 0.05–0.2 s in simulation time;
- environment step: each physics substep or a documented multiple.

All ratios SHOULD be integral in deterministic mode.

### 15.2 Development phase ordering

For each development interval:

1. Latch previous state and external local inputs.
2. Update surface and neighborhood summaries.
3. Update morphogens using double buffers.
4. Update regulatory nodes using double buffers.
5. Evaluate rules into action queues.
6. Resolve action conflicts and energy reservations.
7. Commit death, then division, then mechanical bond edits.
8. Apply property changes, growth, receptor changes, and differentiation.
9. Advance axons; commit neural edge edits.
10. Rebuild dirty indexes and validate invariants.
11. Run configured physics relaxation substeps.

The death-before-division convention MUST be stable and documented.

### 15.3 Runtime/evaluation ordering

For every control interval:

1. Complete all physics substeps from the prior control interval.
2. Aggregate contacts, strain, and environmental samples.
3. Normalize sensors and compute neural input.
4. Advance neural state synchronously.
5. Map neural output to muscle activation.
6. Update muscle bond rest lengths subject to energy availability.
7. For each physics substep:
   1. update environment dynamics;
   2. compute contacts and external fields;
   3. accumulate bond, contact, gravity, and actuator forces/constraints;
   4. integrate/solve;
   5. accumulate damage, sensor statistics, work, and energy costs.
8. Apply energy ledger and starvation consequences.
9. Update evaluator metrics and optional debug snapshot.

No system may read partially updated state from another system unless explicitly declared. Double buffering is required for morphogens, regulatory nodes, and neural activations.

---

## 16. Mutation and evolution hooks

### 16.1 Genome operations

Required mutation operators:

- perturb a bounded scalar using Gaussian or polynomial noise;
- perturb a log-scaled positive parameter;
- change a condition threshold or input weight;
- enable/disable a rule;
- duplicate/delete a rule within configured bounds;
- duplicate/delete a regulatory node and repair references;
- add/remove a condition term;
- add/remove one compatible action;
- change a morphogen/receptor channel reference;
- mutate neural guidance, neural weight, bias, or delay parameters;
- mutate tissue-property deltas.

Crossover MAY be deferred. If implemented, it SHOULD align homologous modules using stable innovation or ancestry identifiers instead of cutting raw serialized bytes.

### 16.2 Mutation validity

After mutation, the genome MUST pass structural validation. Invalid references are repaired deterministically when an unambiguous repair exists; otherwise the mutation is rejected and resampled up to a bounded count. The engine MUST report rejection frequency.

### 16.3 Evaluation contract

```text
EvaluationRequest {
  genome
  experiment_config_hash
  development_seed_set
  environment_seed_set
  replicate_count
}

EvaluationResult {
  fitness_vector
  aggregate_statistics
  per_replicate_outcomes
  viability_flags
  phenotype_descriptors
  artifact_references
}
```

Each genome SHOULD be evaluated across multiple paired seeds. Parent and offspring comparisons SHOULD use the same seed set to reduce noise.

### 16.4 Phenotype descriptors

Record descriptors useful for quality-diversity and analysis:

- cell and bond count;
- tissue composition;
- bounding-box dimensions and symmetry measures;
- number and sizes of connected components;
- neural node/edge count, modularity proxies, and sensor-to-motor reachability;
- passive compliance and active displacement tests;
- locomotion trajectory, duty-cycle, and energy efficiency;
- developmental time, quiescence, and failure reason.

### 16.5 Robustness evaluation

Optional robustness trials SHOULD perturb seed-cell orientation, small initial position noise, morphogen noise, or environment parameters. A robustness objective may reward stable viability and behavior across perturbations. Perturbations must be reproducible and separately seeded.

---

## 17. Determinism and reproducibility

### 17.1 Run identity

Every result MUST record:

- genome content hash;
- experiment configuration hash;
- simulation build/version and serialization schema versions;
- root seed and derived seed policy version;
- deterministic platform profile;
- environment/task version;
- timestep values and solver iteration counts;
- enabled feature flags.

### 17.2 Randomness

Use a counter-based or splittable PRNG. Independent streams MUST exist for at least development, mutation, neural growth, sensor noise, environment, and evaluation perturbations. A code path adding a random draw in one subsystem MUST NOT shift sequences in unrelated subsystems.

### 17.3 Ordering

- Iterate stable IDs or explicitly sorted work lists whenever results can be order-sensitive.
- Use deterministic reduction trees for floating-point sums in the deterministic profile.
- Do not let hash-table iteration order affect simulation results.
- Resolve simultaneous topology changes using the command ordering defined in Section 6.7.
- Parallel scheduling MUST not change committed results.

### 17.4 State hashing and replay

At configurable intervals, compute a canonical state hash over quantized or exact declared fields. A replay compares hashes and reports the first divergent tick and subsystem. Event logs SHOULD be optional because full logs are expensive; checkpoints plus seeds and inputs are the canonical replay mechanism.

### 17.5 Numerical scope

The MVP MUST provide bitwise replay on the same build, hardware class, thread-count policy, and deterministic profile. Cross-platform bitwise identity is a stretch goal. The engine MUST never claim exact reproducibility outside the recorded profile.

---

## 18. Performance and parallelization

### 18.1 Priority order

Optimize in this order:

1. parallelize independent organism evaluations;
2. use data-oriented storage and local spatial indexes within an organism;
3. parallelize per-cell/per-bond read-only calculations;
4. optimize or offload the physics solver only after profiling.

Population-level parallelism is preferred because it preserves simple deterministic organism execution.

### 18.2 Data layout

The production engine SHOULD use structure-of-arrays storage for hot numeric fields and sparse optional components for neural/skin/muscle features. Stable IDs map to packed handles through a generation-checked indirection table. Adjacency lists SHOULD use compact sorted endpoint handles.

### 18.3 Spatial indexing

Use a uniform spatial hash for similarly sized cells; switch to BVH or hierarchical grids if radius ranges become large. Incremental updates SHOULD be used after local movement and division. Neighbor queries MUST return stable-sorted IDs in deterministic mode.

### 18.4 Parallel phases

Safe parallel work includes:

- morphogen edge contributions using deterministic segmented reductions;
- regulatory evaluation per cell;
- force/constraint evaluation with deterministic reduction or graph coloring;
- neural node updates from immutable prior activations;
- sensor sampling;
- metrics over fixed partitions.

Topology edits remain deferred and centrally committed. Avoid unordered atomic floating-point accumulation in the deterministic profile.

### 18.5 Budgets and graceful failure

Configuration MUST bound:

- cells, mechanical bonds, neural nodes/edges, axon tips;
- development ticks and evaluation duration;
- contact pairs and solver iterations;
- memory per organism and debug-trace size;
- rule actions per cell per tick.

When a budget is exceeded, terminate the organism with a typed outcome such as `CELL_LIMIT`, `EDGE_LIMIT`, `TIME_LIMIT`, or `NUMERICAL_FAILURE`. Do not crash the worker.

### 18.6 Baseline performance target

Set a benchmark after the language and solver are selected. A reasonable initial acceptance target is a headless 1,000-cell, 5,000-bond organism with a 500-node/2,000-edge controller completing a 10-second evaluation in no more than 60 seconds on one contemporary CPU core, excluding development and debug tracing. Treat this as a starting target, not a biological scale claim.

---

## 19. Serialization and versioning

### 19.1 Artifact types

The system MUST support:

1. **Genome file:** portable, compact, human-inspectable form available.
2. **Experiment configuration:** all bounds, feature flags, units, task settings, and timing.
3. **Checkpoint:** complete mutable state required to resume exactly.
4. **Phenotype summary:** cells, bonds, neural edges, metrics, lineage, and bounding information for analysis.
5. **Replay manifest:** references genome/config/checkpoints, seeds, build, and expected state hashes.
6. **Event trace:** optional filtered causal/debug events.

### 19.2 Formats

- Use canonical JSON or YAML for small human-edited configs and genome inspection.
- Use a schema-driven binary format such as MessagePack, CBOR, FlatBuffers, or Protobuf for checkpoints and large phenotypes.
- Floating-point text serialization MUST use sufficient precision for round trips.
- Map keys and entity arrays MUST have canonical ordering when hashed.

### 19.3 Version policy

Every artifact includes `schema_version`. Readers MUST reject unknown future major versions with an actionable error. Migrations MUST be explicit, pure, tested, and preserve the original artifact. Simulation semantic changes require a simulator version bump even when the data schema is unchanged.

### 19.4 Checkpoint completeness

A resumable checkpoint includes:

- all cell, bond, neural, environment, energy, evaluator, delay-buffer, and event-queue state;
- stable ID allocators;
- current scheduler counters and fractional accumulators;
- RNG stream/counter state or keys;
- dirty-index state or enough information to rebuild deterministically;
- expected canonical state hash.

---

## 20. Debugging and visualization

### 20.1 Required 3D views

The visual debugger MUST provide:

- cells colored by tissue label, continuous property, lineage, age, damage, rule activation, energy demand, or selected morphogen;
- mechanical bonds colored by strain, stiffness, damage, or muscle influence;
- neural edges colored by weight/sign and animated or shaded by activation;
- axon tips and guidance vectors during development;
- surface status, contact points/normals, and collision shapes;
- morphogen gradient arrows or scalar heat maps;
- environment fields and task landmarks;
- playback pause, single-step, scrub-to-checkpoint, and speed control.

Mechanical and neural graphs MUST have independent visibility toggles.

### 20.2 Cell inspector

Selecting a cell MUST show:

- stable ID, parent, lineage, age, tissue mix, material and functional properties;
- incident mechanical bonds and neural edges;
- morphogen concentrations/gradients and regulatory node state;
- evaluated rule conditions, accepted/rejected actions, and rejection reasons;
- sensor values, neural input/output, motor activation, energy cost, damage;
- recent causality events.

### 20.3 Causal trace

For a configured subset of cells/rules, record:

```text
tick, cell_id, rule_id, input_snapshot_hash,
condition_value, gate_results, emitted_action,
resolution_result, energy_result, topology_event_id
```

Tracing MUST be filterable and bounded. Enabling visualization without tracing MUST not change simulation behavior.

### 20.4 Dashboards

Plot at minimum:

- cell/bond/neural-edge counts over time;
- tissue proportions;
- morphogen min/mean/max;
- total energy and categorized costs;
- center-of-mass trajectory and speed;
- muscle activation/work;
- solver residuals, contact count, damage, and failure events;
- developmental action counts by rule.

### 20.5 Failure report

On failure, save a compact bundle containing run identity, failure enum, last valid checkpoint, first invalid tick, invariant violation, recent causal events, selected metrics, and state hashes.

---

## 21. Validation, tests, and acceptance criteria

### 21.1 Unit tests

Genome and regulatory system:

- Parsing and round-trip serialization preserve canonical genome hashes.
- Every mutation operator returns a valid genome or a typed bounded rejection.
- Rule conditions and conflict resolution match table-driven cases.
- Keyed random gates give identical outputs independent of iteration order.

Morphogens:

- A point source forms a monotonic decaying graph gradient in a symmetric chain/lattice.
- With zero source and decay, total morphogen is conserved within numerical tolerance.
- Symmetric initial conditions remain symmetric.
- Updates are invariant to adjacency-list ordering.

Development:

- Division conserves mass within tolerance and produces nonreused stable IDs.
- Apoptosis removes graph references at the specified commit boundary.
- Cell, edge, and action budgets terminate with correct typed reasons.
- Quiescence detection fires only after the configured consecutive ticks.

Mechanics:

- An isolated spring-damper pair approaches its rest length without energy growth.
- Equal and opposite internal bond forces preserve center-of-mass momentum absent external forces.
- Bone preset deforms less than muscle/skin presets under the same bounded load.
- Bond damage and breakage occur at configured strain history.

Muscle:

- Activation monotonically reduces target rest length within limits.
- Contraction and relaxation obey rate limits.
- A two-anchor muscle fixture produces measurable force/displacement and charges energy.
- Zero energy suppresses or scales contraction according to policy.

Neural:

- Synchronous updates match analytical small-network results.
- Delays arrive on the correct tick.
- Reordering edges does not alter outputs in deterministic mode.
- Axon guidance follows an imposed morphogen gradient and respects path/edge limits.

Sensors and energy:

- Skin contact reports only local configured contacts.
- Proprioceptive output has correct sign and magnitude for known strain.
- Energy ledger equals initial plus income minus categorized costs within tolerance.
- Shortage allocation is proportional within a priority class and order-independent.

Serialization and replay:

- Save/load at a checkpoint produces the same subsequent state-hash sequence.
- Unknown schema versions fail clearly.
- A recorded deterministic profile replays bitwise on its supported platform.

### 21.2 Integration scenarios

1. **Isotropic blob:** one seed with uniform division and adhesion develops a roughly symmetric connected mass.
2. **Gradient axis:** a localized morphogen source creates differentiated zones along a stable axis.
3. **Layered tissue:** surface exposure differentiates skin while an interior morphogen creates bone-like tissue.
4. **Emergent hinge fixture:** two stiff clusters joined by compliant tissue bend under a muscle band without a joint primitive.
5. **Reflex arc:** pressure on a skin region travels through at least one neural edge and activates a muscle region.
6. **Proprioceptive feedback:** stretching a tissue region changes neural input and motor response.
7. **Locomotion seed genome:** a hand-authored genome develops a viable agent whose closed-loop controller produces statistically significant displacement relative to its passive/no-neural control.
8. **Mutation neighborhood:** a batch of small mutations yields a measurable nonzero fraction of viable, phenotypically changed offspring and does not produce engine failures.
9. **Parallel replay:** the same evaluation produces identical state hashes at one and multiple worker settings supported by the deterministic profile.

### 21.3 MVP acceptance criteria

The MVP is accepted when all of the following hold:

- A versioned genome develops from one seed cell into at least 100 cells with spatially distinct bone-, muscle-, skin-, and neuron-like regions, without direct coordinate/body-part instructions.
- Bone-, muscle-, skin-, and neuron-like properties exist and are inspectable; physics dispatch is property-based.
- Mechanical and neural graphs are independently stored, visualized, serialized, and tested.
- A developed muscle region changes bond rest lengths under neural activation and moves a connected stiff region.
- A compliant connector between stiff regions behaves as an articulation without an anatomical joint object.
- Local contact sensing and bond-strain proprioception affect neural activations.
- At least one neural pathway connects a sensor region to a muscle region through the separately grown graph.
- The global energy pool charges categorized costs and can suppress growth or actuation.
- A baseline organism completes a ground locomotion episode and produces fitness plus diagnostics.
- Exact replay passes on the declared deterministic platform profile.
- Budget-exceeding and numerically invalid organisms terminate safely with typed failure artifacts.
- The required inspector, graph overlays, timelines, and causal rule trace are operational.

### 21.4 Evolution-readiness criteria

Before long evolutionary runs:

- Run at least 1,000 random or mutated genomes without simulator process crashes.
- Report proportions for valid genome, completed development, viable phenotype, connected sensor-to-motor path, and completed evaluation.
- Verify no fitness-relevant information leaks through global coordinates, evaluator state, renderer state, or nondeterministic scheduling.
- Demonstrate an automated search can improve a simple objective over random baseline in a reduced fixture, such as muscle-driven beam displacement or directed crawling.
- Archive experiment config, genomes, seeds, build identifier, aggregate metrics, and representative replay manifests.

---

## 22. Phased MVP roadmap

### Phase 0 — Deterministic simulation kernel

Deliver:

- unit/config conventions and versioned schemas;
- stable IDs, packed stores, keyed PRNG streams, fixed scheduler;
- headless runner, state hashing, metrics, typed failure handling;
- basic cell particles, bonds, ground collision, and checkpoint skeleton.

Exit criteria:

- spring, collision, serialization, and deterministic replay tests pass for fixed topology.

### Phase 1 — Developmental body construction

Deliver:

- genome validator and small regulatory network;
- cell growth, division, death, differentiation, bond edits, lineage;
- 3–5 graph-diffused morphogens and local gradient estimation;
- neighbor/surface/mechanical regulatory inputs;
- development limits, quiescence, and action causal trace.

Exit criteria:

- isotropic blob, gradient axis, and layered-tissue scenarios pass and replay.

### Phase 2 — Tissue mechanics and active muscle

Deliver:

- continuous material/functional properties and four required presets;
- stable stiff/compliant tissue mechanics, damage, and optional bending/shape matching;
- muscle-controlled rest lengths, energy-aware actuation, settling phase;
- articulation fixture showing emergent joint-like behavior.

Exit criteria:

- all mechanics and muscle unit tests pass; no `Joint` phenotype primitive exists.

### Phase 3 — Sensors and neural controller

Deliver:

- independent neural nodes/edges, continuous synchronous dynamics, delays;
- morphogen-guided axon tips and synapse compatibility;
- skin pressure plus one environmental-field sensor;
- bond-strain proprioception and muscle motor endpoints;
- neural debugging overlays and selected-node contribution trace.

Exit criteria:

- reflex arc and proprioceptive feedback scenarios pass.

### Phase 4 — Environment, fitness, and viable locomotion

Deliver:

- ground/obstacle/stimulus environment interface;
- task evaluator, fitness vector, trajectory and energy metrics;
- one hand-authored developmental genome producing a closed-loop moving agent;
- passive and controller-disabled ablations.

Exit criteria:

- the locomotion seed genome outperforms its passive control across a fixed seed suite.

### Phase 5 — Evolution harness and scale

Deliver:

- mutation operators, validity/repair pipeline, population evaluation API;
- paired-seed evaluation, phenotype descriptors, lineage metadata;
- population-level parallel execution and resource budgets;
- mutation-neighborhood and crash-resilience campaigns.

Exit criteria:

- an automated search improves a reduced locomotion/actuation objective, and 1,000-genome resilience criteria pass.

### Phase 6 — Robustness and research extensions

Candidate work:

- continuous growth/healing during evaluation;
- local nutrient/oxygen diffusion and evolvable transport;
- richer environmental fields and directional sensing;
- sexual crossover with homology alignment;
- plastic neural weights or spiking neurons;
- ellipsoidal cells, richer extracellular matrices, volume constraints, GPU kernels;
- robustness objectives, curriculum learning, novelty search, and quality diversity.

---

## 23. Recommended implementation defaults

These defaults reduce early ambiguity and may be changed through versioned experiment configuration:

| Decision | MVP default |
|---|---|
| Development lifecycle | Develop, settle, freeze topology, evaluate |
| Cell collision shape | Sphere |
| Morphogen space | Mechanical graph |
| Morphogen channels | 4 |
| Regulatory network | Sparse synchronous continuous nodes |
| Tissue representation | Continuous properties with named presets |
| Neural dynamics | Synchronous leaky continuous activations |
| Axon representation | Growth path during development, graph edge after connection |
| Energy | One global pool per organism |
| Mechanics | Fixed-step spring/constraint particles with damping and contact |
| Deterministic randomness | Counter-based keyed PRNG |
| Parallelism | Across organisms first |
| Genome text format | Canonical JSON |
| Checkpoint format | Versioned schema-driven binary |
| First behavioral task | Ground displacement with energy/damage diagnostics |

---

## 24. Suggested public APIs

```text
validate_genome(genome, limits) -> ValidationReport
mutate_genome(genome, mutation_seed, profile) -> MutationResult

create_organism(genome, seed_spec, run_identity) -> Organism
develop(organism, development_config, environment) -> DevelopmentResult
settle(organism, settle_config, environment) -> SettleResult
evaluate(organism, task, evaluation_config) -> EvaluationResult

step_development(organism, environment)
step_control(organism, environment)
step_physics(organism, environment)

snapshot(organism, detail_level) -> ReadOnlySnapshot
checkpoint(organism, environment, evaluator) -> Checkpoint
resume(checkpoint) -> SimulationSession
replay(manifest) -> ReplayReport
```

All failure-capable APIs SHOULD return typed results rather than throwing for expected invalid-genome or nonviable-phenotype outcomes. Programmer errors and corrupt internal invariants may still fail fast after a failure bundle is written.

---

## 25. Suggested repository organization

```text
src/
  core/             ids, math, units, deterministic RNG, scheduler
  genome/           schema, validation, regulatory VM, mutation
  development/      lifecycle, division, differentiation, lineage
  morphogens/       diffusion and gradients
  topology/         stores, adjacency, command buffers, spatial index
  mechanics/        bonds, constraints, contacts, integration, damage
  tissues/          presets and property blending
  neural/           growth, graph, activations, delays
  sensors/          contact, fields, proprioception, normalization
  energy/           ledger and allocation
  environment/      world interface and baseline environment
  evaluation/       tasks, metrics, phenotype descriptors
  evolution/        population API and operators
  serialization/    schemas, migrations, checkpoints, replay
  observability/    traces, snapshots, failure bundles
  app/              headless CLI and interactive viewer
tests/
  unit/
  integration/
  replay/
  benchmarks/
examples/
  genomes/
  experiments/
  replays/
schemas/
docs/
```

The module names are illustrative. The dependency direction SHOULD keep `core` and data schemas independent of visualization and evolutionary algorithms.

---

## 26. Open research questions

These are intentionally not blockers for Phases 0–3:

1. Does graph diffusion provide sufficient positional information as topology changes, or is an extracellular grid eventually needed?
2. Which mechanics formulation best balances stiff bone-like regions, compliant tissue, determinism, and speed?
3. Should neural edges persist through large tissue motion, retract based on strain, or be represented as mechanically constrained fibers?
4. Which regulatory encoding produces the best local continuity between genotype and phenotype?
5. How much developmental noise improves robustness without obscuring selection?
6. What fitness curricula avoid brittle “falling forward” solutions while not over-prescribing anatomy?
7. When does a global energy pool become an exploitable abstraction that must be replaced by transport?
8. Which symmetry-breaking inputs—seed polarity, initial morphogen asymmetry, gravity, or stochasticity—best support evolvable bilateral agents?

Experiments addressing these questions MUST be isolated by versioned feature flags and configuration hashes.

---

## 27. Definition of done for the first coding-agent implementation

A coding agent implementing this specification is done with the initial research platform only when it has produced:

- a headless deterministic simulator with the module boundaries in Section 5;
- versioned genome/config/checkpoint/replay schemas;
- the four tissue presets implemented through continuous properties;
- local regulatory development, graph morphogens, division, differentiation, death, and topology commands;
- property-based mechanics, active muscle bonds, contact, and emergent-articulation fixture;
- independent neural growth/control graph, skin sensing, and proprioception;
- global energy accounting and baseline environment/evaluator;
- mutation/evaluation hooks suitable for population workers;
- the required visual inspection and causal-debug features;
- passing unit, integration, replay, safety, and baseline performance tests;
- example artifacts for the layered organism, hinge fixture, reflex arc, and locomotion seed genome;
- documentation of any deviation from this spec, including rationale and compatibility impact.

The implementation MUST not substitute a direct body blueprint, named anatomy, hard-coded joints, or scripted gait for the developmental and control mechanisms above.
