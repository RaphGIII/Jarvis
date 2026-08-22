from developer.self_programming import CodeCandidate, Experiment, FitnessResult
from learning.continual.consolidation import MemoryConsolidator
from learning.continual.forgetting import MemoryStrength, compute_forgetting
from learning.continual.learner import ContinualLearner
from learning.curriculum.curriculum import CurriculumManager, TaskCandidate
from learning.curriculum.difficulty import DifficultyEstimator, TaskFeatures
from learning.curriculum.stages import DevelopmentalStage, DevelopmentalStageEvaluator
from learning.experience.transition import Transition
from learning.experience.trajectory import Trajectory
from learning.meta.capability_graph import CapabilityGraph
from learning.meta.metrics import ExponentialMovingAverage
from learning.meta.self_model import SelfModel
from learning.skills.discovery import SkillDiscovery
from learning.skills.option import Option
from learning.world_model.uncertainty import UncertaintyEstimator


def test_ema_and_capability_estimates():
    ema = ExponentialMovingAverage(alpha=0.5)
    assert ema.update(1.0) == 1.0
    assert ema.update(0.0) == 0.5

    self_model = SelfModel()
    estimate = self_model.update_capability("debug_code", success=True, reward=1.0, uncertainty=0.2)
    self_model.update_capability("debug_code", success=False, reward=0.0, uncertainty=0.8)
    assert estimate.attempts == 2
    assert estimate.success_rate == 0.5
    assert self_model.weakest_capabilities()[0].capability == "debug_code"


def test_curriculum_selects_zone_of_proximal_development():
    manager = CurriculumManager(DifficultyEstimator())
    candidates = [
        TaskCandidate("too_easy", TaskFeatures(0.1, 0.1, 0.1, 0.1), predicted_success=0.95),
        TaskCandidate("zpd_harder", TaskFeatures(0.8, 0.7, 0.5, 0.5), predicted_success=0.7),
        TaskCandidate("too_hard", TaskFeatures(1.0, 1.0, 1.0, 1.0), predicted_success=0.2),
    ]
    assert manager.select_next_task(candidates).task_id == "zpd_harder"


def test_forgetting_metrics_and_continual_learner():
    assert compute_forgetting(0.9, 0.4) == 0.5
    assert MemoryStrength(importance=1.0, retrieval_count=2, reward=0.5, age=1.0).score() > 0.0

    learner = ContinualLearner()
    learner.update_benchmark_score("python_debug", 0.9)
    learner.update_benchmark_score("python_debug", 0.6)
    assert learner.forgetting("python_debug") == 0.30000000000000004


def test_uncertainty_calculation():
    estimate = UncertaintyEstimator().combine(
        model_uncertainty=1.0,
        memory_uncertainty=0.5,
        disagreement=0.5,
        novelty=0.0,
    )
    assert 0.0 < estimate.total < 1.0
    assert estimate.action_mode(high=0.4) == "seek_more_information"


def make_action_trajectory(actions, success=True):
    trajectory = Trajectory(metadata={"goal": "test"})
    for index, action in enumerate(actions):
        trajectory.add(
            Transition(
                observation=index,
                latent_state=None,
                action=action,
                reward=1.0,
                next_observation=index + 1,
                next_latent_state=None,
                done=index == len(actions) - 1,
                success=success,
            )
        )
    return trajectory


def test_skill_abstractions_and_memory_consolidation():
    trajectories = [
        make_action_trajectory(["read", "analyze", "test"], success=True),
        make_action_trajectory(["read", "analyze", "test"], success=True),
    ]
    discovered = SkillDiscovery(min_support=2, sequence_length=3).discover(trajectories)
    assert discovered[0].action_sequence == ("read", "analyze", "test")
    option = discovered[0].as_option()
    assert isinstance(option, Option)
    assert option.should_terminate(["read", "analyze", "test"])

    consolidated = MemoryConsolidator(min_repetitions=2).consolidate(trajectories)
    assert consolidated.procedural_candidates[0]["support"] == 2
    assert consolidated.semantic_candidates[0]["samples"] == 2


def test_developmental_stage_evaluation_and_capability_graph():
    evaluator = DevelopmentalStageEvaluator()
    evaluation = evaluator.evaluate(
        {
            "single_tool": 0.6,
            "short_sequences": 0.6,
            "failure_detection": 0.7,
        }
    )
    assert evaluation.stage == DevelopmentalStage.LEARNER
    assert "skill_abstraction" in evaluation.missing_capabilities

    graph = CapabilityGraph()
    graph.add_capability("understand_code", dependencies={"read_file"})
    graph.update("read_file", competence=0.8, confidence=0.9)
    graph.update("understand_code", competence=0.2, confidence=0.5, learning_progress=0.1)
    assert graph.next_training_target().name == "understand_code"


def test_self_programming_fitness_models_are_non_autonomous():
    candidate = CodeCandidate("c1", "improve test", "adds tests")
    result = FitnessResult(correctness=1.0, test_score=0.8, complexity=0.1, risk=0.0)
    experiment = Experiment("exp", candidates=[candidate], fitness_results={"c1": result})
    assert result.effective_fitness() > 1.0
    assert experiment.promote_best() == "c1"
