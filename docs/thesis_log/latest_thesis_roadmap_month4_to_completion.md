# Latest Thesis Roadmap — Month 4 to Completion

## Thesis Title

**Graph Neural Networks for Learning-Based Graph Coloring: An Empirical Study in Sparse Derivative Computation**

## Overall Thesis Focus

The thesis investigates whether a Graph Neural Network-based learned ordering strategy can produce valid graph colorings that are competitive with classical heuristic methods, especially ColPack baselines.

The main experimental pipeline is:

    sparse matrix
    → graph construction
    → heuristic coloring baselines / labels
    → node features
    → GNN
    → node scores
    → vertex ordering
    → greedy coloring
    → comparison against heuristics

The core thesis remains at the graph-coloring level. Sparse Jacobian and Hessian computation provide the motivating application context.

The current main focus is distance-1 graph coloring because this is the safest way to build, validate, and strengthen the full learning-based pipeline. Distance-2 coloring remains the planned Jacobian-related extension and should be introduced after the expanded distance-1 evaluation is stable. Star coloring is kept as a possible Hessian-related optional extension, especially because of its relevance to Hessian sparse derivative computation.

The thesis-safe positioning is:

    Main contribution:
    learned GNN-based vertex ordering for graph coloring.

    Planned extension:
    distance-2 coloring for Jacobian-related sparse derivative computation.

    Optional extension:
    star coloring for Hessian-related sparse derivative computation.

---

# Current Status Before Month 4

Weeks 9--12 completed the first full learning-based prototype.

The completed pipeline is:

    graph
    → node features
    → PyTorch Geometric data
    → GNN node scorer
    → predicted node scores
    → learned vertex ordering
    → greedy coloring
    → learned-vs-heuristic comparison

The first systematic evaluation showed that the learned ordering pipeline works end to end and produces valid colorings.

However, the dataset was still too small for strong generalization claims.

Therefore, the main goal of Month 4 is to expand the dataset and produce a larger distance-1 evaluation before changing the model too much or moving deeply into distance-2 or star coloring.

Distance-2 should not be forgotten. It should be prepared near the end of Month 4 and then started more concretely during Month 5.

---

# Month 4 — Expanded Dataset, Stronger Distance-1 Evaluation, and Distance-2 Preparation

## Month 4 Goal

Strengthen the distance-1 learned-ordering pipeline by expanding the sparse matrix dataset, generating ColPack baselines and ordering targets for the new graphs, rebuilding the PyTorch Geometric dataset, and running a larger learned-versus-heuristic comparison.

At the end of Month 4, the project should also prepare a clear transition toward distance-2 coloring. The goal is not to fully replace the distance-1 work with distance-2, but to make sure the distance-2 extension is technically planned and ready to start.

---

## Week 13 — Dataset Expansion and Graph Selection

### Main Goal

Create a larger and cleaner sparse matrix graph dataset for the next experiments.

The initial five-graph dataset was useful for pipeline validation, but it was too small for reliable generalization analysis. Week 13 therefore focuses on selecting and verifying additional sparse matrix graphs.

### Tasks

- Define graph selection criteria.
- Select candidate sparse matrices from standard sparse matrix benchmark collections.
- Prefer manageable graph sizes.
- Prefer structurally diverse graphs.
- Include some Hessian/star-coloring-friendly matrices where possible.
- Verify each candidate matrix through:
  - Matrix Market loading,
  - graph conversion,
  - vertex count,
  - edge count,
  - duplicate/trivial graph checking.
- Exclude graphs that are invalid, edgeless, too large, or duplicate another graph.
- Generate a summary CSV for the expanded raw graph dataset.
- Document the selection process clearly.
- Keep notes on which graphs may be useful later for distance-2 or star-coloring experiments.

### Accepted New Graphs

The Week 13 expansion accepted the following 10 new graph instances:

- bcsstk01
- bcsstk03
- bcsstk04
- bcsstk05
- bcsstk06
- dwt_234
- dwt_361
- dwt_419
- west0479
- sherman1

The duplicate candidate `bcsstk07` was rejected because it produced the same graph structure as `bcsstk06` under the current sparsity-pattern graph conversion.

### Final Week 13 Dataset Size

    original graphs: 5
    accepted new graphs: 10
    total graphs: 15

### Week 13 Deliverables

- `docs/thesis_log/week13_dataset_expansion_plan.md`
- `docs/thesis_log/week13_candidate_matrices.md`
- `src/training/build_week13_expanded_graph_summary.py`
- `data/processed/initial_graph_coloring_dataset/graph_metadata/week13_expanded_graph_summary.csv`
- 10 accepted new `.mtx` files in `data/raw/matrices/`

### Week 13 Outcome Statement

By the end of Week 13, the initial five-graph dataset was expanded into a verified 15-graph sparse matrix dataset. The new graphs were selected based on compatibility, manageable size, structural diversity, ordering relevance, and possible Hessian/star-coloring readiness. Invalid, trivial, and duplicate candidates were excluded. This expanded dataset provides the foundation for generating ColPack baselines and ordering targets in Week 14.

---

## Week 14 — ColPack Baselines and Ordering Targets

### Main Goal

Generate distance-1 ColPack baseline results and ordering targets for the expanded 15-graph dataset.

### Tasks

- Prepare the final expanded graph list for ColPack execution.
- Run ColPack distance-1 coloring on all accepted graphs.
- Use at least the following orderings:
  - `SMALLEST_LAST`
  - `LARGEST_FIRST`
- Save ColPack output files for every graph and ordering.
- Parse ColPack outputs into a structured benchmark table.
- Extract actual vertex orderings from ColPack outputs.
- Build normalized node-level ordering targets.
- Check that each target aligns with the corresponding graph vertices.
- Identify graphs where different orderings produce different color counts.
- Record any graph where ColPack output and Python graph construction disagree.

### Important Checks

For every graph:

    ColPack vertex count == Python graph vertex count
    ColPack edge count == Python graph edge count
    ordering contains every vertex exactly once
    target length == number of graph vertices
    coloring is valid

### Deliverables

- Expanded ColPack output files.
- Updated ColPack benchmark CSV.
- Expanded ordering target CSV.
- Notes on ordering-sensitive graphs.
- Notes on any failed or problematic graph.

### Week 14 Outcome Statement

By the end of Week 14, every graph in the expanded dataset should have verified distance-1 ColPack baseline results and normalized ordering targets. This prepares the dataset for rebuilding the PyTorch Geometric graph files and rerunning the GNN experiments.

---

## Week 15 — Rebuild PyG Dataset and Rerun GNN Experiments

### Main Goal

Update the learning dataset and rerun the current GNN pipeline on the expanded graph set.

### Tasks

- Update graph metadata.
- Create a new graph-level train/validation/test split.
- Rebuild node features for all 15 graphs.
- Attach normalized ordering targets to each graph.
- Save updated PyTorch Geometric `.pt` files.
- Check feature and target consistency.
- Train the existing GNN node scorer on the expanded dataset.
- Run repeated training with multiple random seeds.
- Save checkpoints, losses, predictions, and learned orderings.

### Feature Policy

Do not overcomplicate the feature set yet.

Start with the current stable features:

- degree
- normalized degree
- clustering coefficient
- core number
- constant bias

Only test extra features after the expanded simple pipeline works.

Possible later features include:

- triangle count
- average neighbor degree
- component size
- graph-normalized structural features

### Deliverables

- Updated PyTorch Geometric dataset.
- Updated graph split file.
- Updated feature summary.
- Repeated GNN training logs.
- Saved model checkpoints.
- Initial learned-ordering outputs on the expanded dataset.

### Week 15 Outcome Statement

By the end of Week 15, the GNN training and prediction pipeline should run on the expanded 15-graph dataset, producing learned orderings for multiple validation and test graphs.

---

## Week 16 — Larger Learned-vs-Heuristic Comparison and Distance-2 Preparation Checkpoint

### Main Goal

Evaluate whether the learned ordering is competitive with ColPack and NetworkX baselines on the expanded distance-1 dataset, and prepare the technical transition toward distance-2 coloring.

The main output of Week 16 is still the larger distance-1 learned-versus-heuristic comparison. However, Week 16 should also include a distance-2 preparation checkpoint so that the distance-2 extension is not delayed too far.

### Distance-1 Evaluation Tasks

- Convert predicted node scores into learned vertex orderings.
- Apply greedy coloring using learned orderings.
- Verify all learned colorings.
- Compare learned results against:
  - ColPack `SMALLEST_LAST`
  - ColPack `LARGEST_FIRST`
  - NetworkX `largest_first`
  - NetworkX `smallest_last`
  - NetworkX `random_sequential`
- Analyze color counts.
- Analyze runtime.
- Analyze seed stability.
- Separate ordering quality from coloring quality.
- Identify graph cases where:
  - GNN matches heuristics,
  - GNN is worse than heuristics,
  - GNN possibly improves,
  - all methods behave similarly.

### Distance-2 Preparation Checkpoint

At the end of Week 16, the project should also prepare for the distance-2 extension.

This does not require full distance-2 implementation yet. The goal is to make sure the next extension step is technically clear.

Preparation tasks:

- Review the existing distance-2 preparation notes.
- Identify which expanded graphs are suitable for first distance-2 experiments.
- Check which scripts need to be extended for `coloring_distance = 2`.
- Inspect whether the ColPack runner needs changes so that coloring distance can be passed as an argument.
- Decide whether distance-2 should start with ColPack baselines only or also include learned-ordering evaluation.
- Write a short distance-2 transition plan.

### Deliverables

- Combined learned-vs-heuristic distance-1 result table.
- Runtime table.
- Seed stability table.
- Graph-wise comparison notes.
- Month 4 result summary.
- `docs/thesis_log/distance2_transition_plan.md`

### Week 16 Outcome Statement

By the end of Week 16, the thesis should have a larger and more meaningful distance-1 learned-versus-heuristic comparison. In addition, the next distance-2 implementation step should be clearly planned, so that Month 5 can begin distance-2 work without disrupting the main distance-1 thesis contribution.

---

# Month 5 — Implementation and Writing in Parallel

## Month 5 Goal

Month 5 should combine final experiments with early thesis writing.

The purpose is to avoid leaving all writing until the final month. Writing should begin with stable parts of the thesis, especially methodology, dataset construction, baseline generation, and the GNN pipeline design.

Implementation should continue, but no large new direction should be introduced unless the main distance-1 results are already stable.

Distance-2 work should begin in Month 5 in a controlled way. The goal is to add a meaningful Jacobian-related extension without weakening the main learned-ordering contribution.

---

## Week 17 — Main Distance-1 Experiment Batch, Begin Distance-2 Baseline Setup, and Methodology Writing

### Implementation Goal

Run the final or near-final distance-1 experiment batch on the expanded dataset and begin the technical setup for distance-2 baseline experiments.

### Distance-1 Implementation Tasks

- Finalize distance-1 experiment setup.
- Finalize graph splits.
- Finalize baseline methods.
- Rerun main distance-1 experiments if needed.
- Clean result tables.
- Save experiment configurations and logs.

### Distance-2 Setup Tasks

- Review the distance-2 transition plan from Week 16.
- Extend or create a distance-2 validity checker.
- Decide how distance-2 graph constraints will be verified.
- Check whether existing benchmark schemas already support `coloring_distance = 2`.
- Prepare selected graphs for initial distance-2 testing.
- Start with a small subset before running all graphs.

### Writing Goal

Start writing the methodology chapter.

### Writing Sections

- Graph coloring problem definition.
- Sparse derivative motivation.
- Sparse matrix to graph conversion.
- ColPack baseline generation.
- Node feature extraction.
- GNN node-scoring model.
- Learned ordering pipeline.
- Evaluation metrics.
- Brief explanation of why distance-1 is implemented first and distance-2 is treated as an extension.

### Week 17 Outcome Statement

By the end of Week 17, the thesis should have a stable main distance-1 experiment setup, the first technical pieces for distance-2 baseline evaluation, and the first draft of the methodology chapter.

---

## Week 18 — Distance-1 Comparative Analysis, Distance-2 ColPack Baselines, and Dataset/Experimental Setup Writing

### Implementation Goal

Analyze the graph-wise distance-1 results and run the first distance-2 ColPack baseline experiments on selected graphs.

### Distance-1 Analysis Tasks

- Compare learned method with ColPack baselines.
- Compare learned method with NetworkX baselines.
- Analyze color-count gaps.
- Analyze behavior by graph size and density.
- Inspect where learned ordering works well.
- Inspect where learned ordering performs poorly.
- Identify whether the validation set predicts test behavior reasonably.

### Distance-2 Baseline Tasks

- Run ColPack distance-2 coloring on a selected subset of graphs.
- Start with manageable graphs before testing larger ones.
- Verify distance-2 colorings using the distance-2 validity checker.
- Store distance-2 results separately from distance-1 results.
- Compare distance-1 and distance-2 color counts for selected graphs.
- Document any ColPack runner changes needed for distance-2.

### Writing Goal

Write the dataset and experimental setup sections.

### Writing Sections

- Original dataset.
- Week 13 expanded dataset.
- Graph statistics.
- Candidate selection and verification process.
- Train/validation/test split.
- Baseline methods.
- GNN training setup.
- Repeated random seeds.
- Runtime measurement setup.
- Initial distance-2 baseline setup, if completed.

### Week 18 Outcome Statement

By the end of Week 18, the thesis should have a clear experimental setup draft, a first interpretation of the expanded distance-1 results, and initial verified distance-2 baseline results on selected graphs.

---

## Week 19 — Robustness/Ablations, Optional Learned Distance-2 Evaluation, and Results Writing

### Implementation Goal

Run small controlled robustness checks for the learned distance-1 pipeline and, if the distance-2 baseline workflow is stable, test whether the learned ordering can be reused for distance-2 evaluation.

### Distance-1 Robustness Tasks

- Check random seed robustness.
- Run small feature ablations:
  - degree only,
  - degree + normalized degree,
  - current full feature set.
- Optionally test small model variations:
  - one GNN layer,
  - two GNN layers.
- Avoid large redesigns unless clearly necessary.
- Clean result folders and scripts.

### Optional Learned Distance-2 Tasks

Only do these if the distance-2 baseline workflow from Week 18 is stable.

- Reuse distance-1 learned orderings for distance-2 greedy coloring.
- Compare learned-ordering distance-2 results against ColPack distance-2 baselines.
- Verify all distance-2 colorings carefully.
- Record color counts and runtime.
- Document whether separate distance-2 training may be needed later.

### Writing Goal

Start writing the results chapter.

### Writing Sections

- ColPack distance-1 baseline results.
- NetworkX distance-1 baseline results.
- GNN learned-ordering distance-1 results.
- Learned-vs-heuristic distance-1 comparison.
- Runtime results.
- Seed stability results.
- Feature/model ablation results, if completed.
- Initial distance-2 baseline results, if completed.
- Optional learned distance-2 results, if completed.

### Week 19 Outcome Statement

By the end of Week 19, the thesis should have supporting evidence about robustness, a first draft of the results chapter, and possibly an initial learned-ordering distance-2 evaluation if the baseline workflow is stable.

---

## Week 20 — Distance-2 Summary, Optional Star-Coloring Preparation, and Discussion Writing

### Implementation Goal

Summarize the distance-2 work and prepare the optional star-coloring direction without risking the main thesis.

### Distance-2 Summary Tasks

- Summarize distance-2 baseline results.
- Compare distance-1 and distance-2 behavior on selected graphs.
- Decide whether distance-2 learned-ordering evaluation is strong enough to include as a result or should be discussed as preliminary.
- Document limitations of the distance-2 extension.
- Save distance-2 result tables and notes.

### Optional Star-Coloring Preparation Tasks

Star coloring is considered only as an optional Hessian-related extension.

Preparation tasks:

- Identify Hessian/star-coloring-ready graphs.
- Focus on:
  - `hess_pat`
  - `hess_pat_small`
  - `bcsstk*` matrices
- Summarize how star coloring relates to Hessian direct recovery.
- Use the ADIC2/ColPack background paper as motivation.
- Avoid implementing full star coloring unless the main thesis results are already stable and time allows.

### Writing Goal

Start writing the discussion chapter.

### Writing Sections

- What the learned method achieved.
- Where it matched heuristics.
- Where heuristics remained stronger.
- Why exact ordering imitation may not be necessary for matching color count.
- Limitations of the current dataset and model.
- Relation to sparse Jacobian/Hessian computation.
- Distance-2 results or distance-2 limitations.
- Star coloring as a possible Hessian-related extension.
- Future work.

### Week 20 Outcome Statement

By the end of Week 20, the thesis should have a clear distance-2 summary or preliminary extension result, an optional star-coloring preparation note, and the discussion chapter should be started.

---

# Month 6 — Thesis Writing, Polishing, and Finalization

## Month 6 Goal

Complete the thesis manuscript, polish the experiments and repository, and prepare the final submission package.

Implementation during Month 6 should be limited to small fixes, missing reruns, or table regeneration. No major new experimental direction should be introduced during Month 6.

---

## Week 21 — Complete Methodology and Experimental Setup

### Writing Tasks

- Complete methodology chapter.
- Complete dataset description.
- Complete experimental setup section.
- Add or finalize pipeline diagram.
- Check terminology consistency.
- Check notation consistency.
- Integrate distance-2 methodology only if the implementation is complete enough.
- Mention star coloring only as optional/future work unless actually implemented.

### Implementation Tasks

- Minor fixes only.
- Regenerate missing tables if needed.
- Confirm all scripts are reproducible.

### Week 21 Outcome Statement

By the end of Week 21, the methodology and experimental setup chapters should be complete in draft form.

---

## Week 22 — Complete Results Chapter

### Writing Tasks

- Finalize result tables.
- Write distance-1 comparison analysis.
- Write runtime analysis.
- Write seed stability analysis.
- Write feature/model robustness section if available.
- Write distance-2 results section if the experiments are stable enough.
- Add figure and table captions.

### Implementation Tasks

- Rerun only missing or broken experiments.
- Clean result CSVs and logs.

### Week 22 Outcome Statement

By the end of Week 22, the results chapter should be complete in draft form.

---

## Week 23 — Complete Discussion, Conclusion, and Future Work

### Writing Tasks

- Complete discussion chapter.
- Complete conclusion.
- Write limitations section.
- Write future work section.
- Discuss distance-2 extension.
- Discuss optional star-coloring extension for Hessian matrices.
- Connect the findings back to sparse derivative computation.

### Implementation Tasks

- No major implementation.
- Only small corrections if needed.

### Week 23 Outcome Statement

By the end of Week 23, the complete thesis manuscript should exist in draft form.

---

## Week 24 — Final Editing and Submission Preparation

### Final Tasks

- Formatting.
- Bibliography.
- Cross-references.
- Figure captions.
- Table captions.
- Proofreading.
- Consistency checks.
- Repository cleanup.
- Final experiment archive.
- Final supervisor review preparation.
- Final submission package.

### Week 24 Outcome Statement

By the end of Week 24, the thesis should be ready for final supervisor review and submission.

---

# Extension Positioning

## Distance-2 Coloring

Distance-2 coloring remains the main planned extension because it is already part of the original thesis proposal and is directly connected to sparse Jacobian computation.

It should not distract from the expanded distance-1 learned-ordering results, but it should begin after the main distance-1 pipeline is stable.

The planned path is:

    Week 16:
    distance-2 transition plan

    Week 17:
    distance-2 validity checker and baseline setup

    Week 18:
    first ColPack distance-2 baselines on selected graphs

    Week 19:
    optional learned-ordering distance-2 evaluation

    Week 20:
    distance-2 summary and interpretation

## Star Coloring

Star coloring is kept as an optional Hessian-related extension.

It is relevant because ColPack uses different coloring models for sparse derivative computation, and star coloring is associated with Hessian direct recovery.

However, star coloring should not replace the main thesis contribution unless there is enough time and the main distance-1 and distance-2 results are already stable.

The star-coloring direction should be positioned as:

    Optional Hessian-related extension.
    Motivated by sparse Hessian computation.
    Considered after distance-1 and distance-2 work are stable.
    Not required for the main thesis contribution unless time allows.

## Final Thesis-Safe Positioning

    The main contribution is the learned-ordering pipeline for graph coloring.
    Distance-1 is the main validated experimental setting.
    Distance-2 is the planned Jacobian-related extension.
    Star coloring is a possible Hessian-related extension.

---

# High-Level Tracking Table

| Week | Implementation Focus | Writing Focus | Status |
|---|---|---|---|
| Week 13 | Dataset expansion and graph selection | Notes only | In progress / almost complete |
| Week 14 | Distance-1 ColPack baselines and ordering targets | Notes only | Planned |
| Week 15 | Rebuild PyG dataset and rerun GNN | Notes only | Planned |
| Week 16 | Larger distance-1 comparison and distance-2 transition plan | Month 4 summary | Planned |
| Week 17 | Main distance-1 experiments and distance-2 baseline setup | Methodology chapter starts | Planned |
| Week 18 | Distance-1 analysis and first distance-2 ColPack baselines | Dataset and experimental setup | Planned |
| Week 19 | Robustness checks and optional learned distance-2 evaluation | Results chapter starts | Planned |
| Week 20 | Distance-2 summary and optional star-coloring preparation | Discussion chapter starts | Planned |
| Week 21 | Minor fixes only | Methodology/setup complete | Planned |
| Week 22 | Missing reruns only | Results complete | Planned |
| Week 23 | No major implementation | Discussion/conclusion complete | Planned |
| Week 24 | Repository cleanup | Final editing/submission | Planned |

---

# Practical Working Rule

The work should proceed one step at a time.

For each week:

1. Finish the smallest necessary technical step.
2. Inspect the output carefully.
3. Fix problems before moving forward.
4. Document the result.
5. Commit only after the week or major milestone is clean.

This keeps the thesis implementation controlled, reproducible, and meaningful.

---

# Final Roadmap Summary

    Week 13:
    Expand and verify dataset.

    Week 14:
    Generate distance-1 ColPack baselines and ordering targets.

    Week 15:
    Rebuild PyG dataset and rerun GNN.

    Week 16:
    Larger distance-1 learned-vs-heuristic comparison and distance-2 transition plan.

    Week 17:
    Main distance-1 experiments, begin distance-2 baseline setup, and start methodology writing.

    Week 18:
    Distance-1 comparative analysis, first distance-2 ColPack baselines, and write dataset/setup.

    Week 19:
    Robustness checks, optional learned distance-2 evaluation, and write results.

    Week 20:
    Distance-2 summary, optional star-coloring preparation, and write discussion.

    Week 21:
    Complete methodology and experimental setup.

    Week 22:
    Complete results.

    Week 23:
    Complete discussion and conclusion.

    Week 24:
    Final editing and submission preparation.